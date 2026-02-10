"""
LIVE NSE OPTION SIGNAL GENERATOR (SIGNAL-ONLY)

Uses a pre-trained ML model to generate live option trading signals
from Upstox 5-minute NIFTY 50 data.

IMPORTANT:
- Signal-only (NO ORDER PLACEMENT)
- Strategy is locked at start (BUYING or SELLING)
- Designed to avoid overtrading
"""

import os
import time
import sys
import signal
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIGURATION
# =========================

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
if not UPSTOX_ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN not set in environment variables")

BASE_URL = "https://api.upstox.com/v2"

INSTRUMENT_KEY_NIFTY = "NSE_INDEX|Nifty 50"

MODEL_PATH = "logistic_regression_model.pkl"
SCALER_PATH = "logistic_regression_scaler.pkl"
FEATURES_PATH = "logistic_regression_features.pkl"

PROB_THRESHOLD_BUY = 0.65
PROB_THRESHOLD_SELL_NEUTRAL_LOW = 0.45
PROB_THRESHOLD_SELL_NEUTRAL_HIGH = 0.55

MAX_TRADES_PER_DAY = 3
COOLDOWN_MINUTES = 30
NO_TRADE_AFTER = datetime.strptime("15:00", "%H:%M").time()

KILL_SWITCH = False

# =========================
# GLOBAL STATE
# =========================

ACTIVE_STRATEGY = None
TRADES_TAKEN = 0
LAST_TRADE_TIME = None
RUNNING = True

# =========================
# UTILS
# =========================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def graceful_exit(signum, frame):
    global RUNNING
    log("Kill signal received. Shutting down safely.")
    RUNNING = False

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# =========================
# STEP 1: STRATEGY SELECTION
# =========================

def select_strategy():
    global ACTIVE_STRATEGY
    choice = input("Select strategy (BUYING / SELLING): ").strip().upper()
    if choice not in ("BUYING", "SELLING"):
        raise ValueError("Invalid strategy selection")
    ACTIVE_STRATEGY = choice
    log(f"ACTIVE STRATEGY LOCKED: {ACTIVE_STRATEGY}")

# =========================
# STEP 2: LIVE DATA FETCH
# =========================

def fetch_live_data():
    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
    }

    params = {
        "instrument_key": INSTRUMENT_KEY_NIFTY,
        "interval": "5minute",
        "count": 100
    }

    resp = requests.get(
        f"{BASE_URL}/historical-candle/intraday",
        headers=headers,
        params=params,
        timeout=10
    )

    if resp.status_code != 200:
        log(f"Upstox API error: {resp.text}")
        return None

    data = resp.json()["data"]["candles"]
    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df

# =========================
# STEP 3: FEATURE ENGINEERING
# =========================

def create_live_features(df):
    df = df.copy()

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["distance_from_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    df["volatility_20"] = df["log_return_1"].rolling(20).std() * np.sqrt(252)

    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    df["bb_position"] = (df["close"] - lower) / (upper - lower)

    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["day_of_week"] = df.index.dayofweek
    df["hour"] = df.index.hour

    df["candle_body"] = (df["close"] - df["open"]) / df["open"]
    df["candle_range"] = (df["high"] - df["low"]) / df["open"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_ratio"] = tr / atr

    return df.dropna()

# =========================
# STEP 4: MODEL LOADING
# =========================

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

# =========================
# STEP 5: SIGNAL GENERATION
# =========================

def generate_signal(prob, row):
    trend_up = row["momentum_5"] > 0 and row["rsi_14"] > 50
    trend_down = row["momentum_5"] < 0 and row["rsi_14"] < 50
    vol_expanding = row["atr_ratio"] > 1.1

    if ACTIVE_STRATEGY == "BUYING":
        if prob > PROB_THRESHOLD_BUY and trend_up and vol_expanding:
            return "BUY CALL", "High probability breakout with momentum"
        if prob < (1 - PROB_THRESHOLD_BUY) and trend_down:
            return "BUY PUT", "Bearish momentum continuation"
        return "NO TRADE", "Conditions not aligned"

    if ACTIVE_STRATEGY == "SELLING":
        if PROB_THRESHOLD_SELL_NEUTRAL_LOW <= prob <= PROB_THRESHOLD_SELL_NEUTRAL_HIGH and vol_expanding:
            if trend_up:
                return "SELL CALL", "Mean reversion with elevated volatility"
            if trend_down:
                return "SELL PUT", "Downside exhaustion with high IV"
        return "NO TRADE", "Seller conditions not met"

# =========================
# STEP 6: OPTION SELECTION
# =========================

def select_atm_option(spot_price, option_type):
    strike = round(spot_price / 50) * 50
    expiry = "NEAREST_WEEKLY"
    return strike, expiry, option_type

# =========================
# STEP 7: RISK FILTERS
# =========================

def apply_risk_filters():
    global TRADES_TAKEN, LAST_TRADE_TIME

    now = datetime.now()

    if KILL_SWITCH:
        return False, "Kill switch active"

    if now.time() >= NO_TRADE_AFTER:
        return False, "Post cutoff time"

    if TRADES_TAKEN >= MAX_TRADES_PER_DAY:
        return False, "Max trades reached"

    if LAST_TRADE_TIME and (now - LAST_TRADE_TIME) < timedelta(minutes=COOLDOWN_MINUTES):
        return False, "Cooldown active"

    return True, None

# =========================
# MAIN LOOP
# =========================

def main_loop():
    global TRADES_TAKEN, LAST_TRADE_TIME

    model, scaler, features = load_model()
    log("Model and scaler loaded")

    while RUNNING:
        df = fetch_live_data()
        if df is None:
            time.sleep(30)
            continue

        feat_df = create_live_features(df)
        if feat_df.empty:
            time.sleep(30)
            continue

        latest = feat_df.iloc[-1]
        X = pd.DataFrame([latest[features]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]

        allowed, reason = apply_risk_filters()
        signal_text = "NO TRADE"
        explanation = reason or "Risk filter passed"

        if allowed:
            signal_text, explanation = generate_signal(prob, latest)
            if signal_text != "NO TRADE":
                TRADES_TAKEN += 1
                LAST_TRADE_TIME = datetime.now()

        strike, expiry, opt_type = select_atm_option(latest["close"], signal_text)

        log(
            f"SIGNAL | {ACTIVE_STRATEGY} | {signal_text} | "
            f"Prob={prob:.3f} | Strike={strike} | Expiry={expiry} | Reason={explanation}"
        )

        time.sleep(300)

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    select_strategy()
    main_loop()
