"""
ORION-NSE : LIVE OPTION SIGNAL ENGINE (SIGNAL ONLY)

IMPORTANT SAFETY GUARANTEES:
- NO order placement
- NO trading endpoints
- READ-ONLY market data
- User must choose strategy at startup
- Strategy locked for entire session

This script:
- Uses NIFTY INDEX only for ATM strike discovery
- Trades are SIGNALS on NIFTY OPTIONS
- ML model is used as a FILTER, not a predictor (temporary compromise)
"""

# ============================
# STANDARD IMPORTS
# ============================

import os
import sys
import time
import math
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ============================
# ENV SETUP
# ============================

load_dotenv()

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
if not UPSTOX_ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN not set in environment variables")

BASE_URL = "https://api.upstox.com/v2"
HEADERS = {
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
    "Accept": "application/json"
}

# ============================
# USER STRATEGY SELECTION
# ============================

def select_strategy():
    choice = input("Select strategy (BUYING / SELLING): ").strip().upper()
    if choice not in {"BUYING", "SELLING"}:
        raise ValueError("Invalid strategy selection")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ACTIVE STRATEGY LOCKED: {choice}")
    return choice

# ============================
# MODEL LOADING
# ============================

def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model and scaler loaded")
    return model, scaler, features

# ============================
# INDEX QUOTE (REFERENCE ONLY)
# ============================

def get_nifty_ltp():
    url = f"{BASE_URL}/market-quote/quotes"
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    return data["NSE_INDEX|Nifty 50"]["last_price"]

# ============================
# OPTION SELECTION
# ============================

def nearest_weekly_expiry():
    today = datetime.now().date()
    # NSE weekly expiry is Thursday
    days_ahead = (3 - today.weekday()) % 7
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime("%d%b%y").upper()

def round_to_50(price):
    return int(round(price / 50) * 50)

def option_instrument(strike, expiry, opt_type):
    return f"NSE_FO|NIFTY{expiry}{strike}{opt_type}"

# ============================
# OPTION CANDLE FETCH
# ============================

def fetch_option_candles(instrument_key, minutes=60):
    end = datetime.now()
    start = end - timedelta(minutes=minutes)

    url = f"{BASE_URL}/historical-candle/intraday"
    params = {
        "instrument_key": instrument_key,
        "interval": "5minute",
        "from": start.strftime("%Y-%m-%d %H:%M"),
        "to": end.strftime("%Y-%m-%d %H:%M")
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    r.raise_for_status()

    candles = r.json()["data"]["candles"]
    if not candles or len(candles) < 6:
        return None

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df

# ============================
# FEATURE MAPPING (TEMPORARY)
# ============================

def create_live_features(df):
    """
    We deliberately reuse the OLD feature logic.
    Option OHLC is treated like underlying OHLC.
    Model acts only as a FILTER.
    """

    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["momentum"] = df["close"] - df["close"].shift(3)
    df["volatility"] = df["return"].rolling(5).std()

    df = df.dropna()
    if df.empty:
        return None

    return df.iloc[-1]

# ============================
# SIGNAL GENERATION
# ============================

def generate_signal(
    strategy,
    option_type,
    prob,
    opt_row
):
    """
    ML probability is only a GATE.
    Strategy logic does the real work.
    """

    price_change = opt_row["momentum"]
    vol = opt_row["volatility"]

    if strategy == "BUYING":
        if prob > 0.65 and price_change > 0 and vol > 0.01:
            return f"BUY {option_type}"
    else:
        if 0.45 < prob < 0.55 and vol < 0.01:
            return f"SELL {option_type}"

    return "NO TRADE"

# ============================
# RISK FILTERS
# ============================

MAX_SIGNALS_PER_DAY = 3
COOLDOWN_MINUTES = 30
NO_TRADE_AFTER = datetime.strptime("15:00", "%H:%M").time()

# ============================
# MAIN LOOP
# ============================

def main_loop():
    strategy = select_strategy()
    model, scaler, feature_names = load_model()

    trades_today = 0
    last_signal_time = None

    while True:
        now = datetime.now()

        if now.time() > NO_TRADE_AFTER:
            print("Trading window closed.")
            break

        if trades_today >= MAX_SIGNALS_PER_DAY:
            print("Max signals reached for the day.")
            break

        if last_signal_time and (now - last_signal_time).seconds < COOLDOWN_MINUTES * 60:
            time.sleep(30)
            continue

        try:
            nifty_ltp = get_nifty_ltp()
            atm = round_to_50(nifty_ltp)
            expiry = nearest_weekly_expiry()

            ce = option_instrument(atm, expiry, "CE")
            pe = option_instrument(atm, expiry, "PE")

            for opt_key, opt_type in [(ce, "CALL"), (pe, "PUT")]:
                df = fetch_option_candles(opt_key)
                if df is None:
                    continue

                features = create_live_features(df)
                if features is None:
                    continue

                X = features[feature_names].values.reshape(1, -1)
                X_scaled = scaler.transform(X)
                prob = model.predict_proba(X_scaled)[0][1]

                signal = generate_signal(strategy, opt_type, prob, features)

                print(
                    f"[{now.strftime('%H:%M:%S')}] "
                    f"{signal} | {opt_key} | Prob={prob:.3f}"
                )

                if signal != "NO TRADE":
                    trades_today += 1
                    last_signal_time = now

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(300)  # 5-minute cycle

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    main_loop()
