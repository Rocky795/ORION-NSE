"""
ORION-NSE : LIVE OPTION SIGNAL ENGINE (SIGNAL ONLY)

SAFETY GUARANTEES:
- NO order placement
- NO trading endpoints
- READ-ONLY market data
- Strategy locked for entire session
"""
import json
import os
import time
import joblib
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ============================
# ENV SETUP
# ============================

load_dotenv()

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
if not UPSTOX_ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN not set")

BASE_URL_V2 = "https://api.upstox.com/v2"

HEADERS = {
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ============================
# STRATEGY SELECTION
# ============================

def select_strategy():
    choice = input("Select strategy (BUYING / SELLING): ").strip().upper()
    if choice not in ("BUYING", "SELLING"):
        raise ValueError("Invalid strategy")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ACTIVE STRATEGY LOCKED: {choice}")
    return choice

# ============================
# LOAD MODEL
# ============================

def load_model():
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("logistic_regression_scaler.pkl")
    features = joblib.load("logistic_regression_features.pkl")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model and scaler loaded")
    return model, scaler, features

# ============================
# NIFTY LTP (REFERENCE ONLY)
# ============================

def get_nifty_ltp():
    url = f"{BASE_URL_V2}/market-quote/quotes"
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}

    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    r.raise_for_status()

    for v in r.json().get("data", {}).values():
        if "last_price" in v:
            return v["last_price"]

    raise RuntimeError("Failed to fetch NIFTY LTP")

# ============================
# OPTION CHAIN
# ============================


def fetch_option_chain(expiry_date):
    url = f"{BASE_URL_V2}/option/chain"
    params = {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "expiry_date": expiry_date
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    r.raise_for_status()

    return r.json().get("data", [])



def normalize_expiry(exp):
    if isinstance(exp, int):
        return datetime.fromtimestamp(exp).strftime("%Y-%m-%d")
    if isinstance(exp, str):
        return exp
    return None


def get_valid_nifty_expiries():
    if not os.path.exists("NSE.json"):
        raise RuntimeError("Instrument master file NSE.json not found")

    with open("NSE.json", "r") as f:
        instruments = json.load(f)

    expiries = set()

    for inst in instruments:
        if inst.get("underlying_symbol") != "NIFTY":
            continue

        raw_exp = inst.get("expiry")
        if not raw_exp:
            continue

        norm = normalize_expiry(raw_exp)
        if norm:
            expiries.add(norm)

    if not expiries:
        raise RuntimeError("No NIFTY expiries found in instrument master")

    return sorted(expiries)


def get_nearest_valid_expiry():
    today = datetime.now().date()

    for exp in get_valid_nifty_expiries():
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        if exp_date >= today:
            return exp

    raise RuntimeError("No future NIFTY expiry available")




# ============================
# FEATURE PROXY (TEMPORARY)
# ============================

def build_feature_vector(opt):
    """
    Proxy features to satisfy old ML model.
    ML is used only as a FILTER.
    """

    md = opt["market_data"]
    og = opt["option_greeks"]

    return {
        "return": 0.0,
        "range": (md["ask_price"] - md["bid_price"]) / max(md["ltp"], 1),
        "momentum": md["ltp"] - md["close_price"],
        "volatility": og["iv"] / 100.0,
    }

# ============================
# SIGNAL LOGIC
# ============================

def generate_signal(strategy, prob, greeks):
    if strategy == "BUYING":
        if prob > 0.65 and abs(greeks["delta"]) > 0.4 and greeks["iv"] > 15:
            return "BUY"
    else:
        if 0.45 < prob < 0.55 and greeks["theta"] < -5 and greeks["iv"] > 20:
            return "SELL"
    return "NO TRADE"

# ============================
# MAIN LOOP
# ============================

MAX_SIGNALS_PER_DAY = 3
COOLDOWN_MINUTES = 30
NO_TRADE_AFTER = datetime.strptime("15:00", "%H:%M").time()

def main_loop():
    strategy = select_strategy()
    model, scaler, feature_names = load_model()

    trades_today = 0
    last_signal = None

    while True:
        now = datetime.now()

        if now.time() > NO_TRADE_AFTER:
            print("Trading window closed.")
            break

        if trades_today >= MAX_SIGNALS_PER_DAY:
            print("Max signals reached.")
            break

        if last_signal and (now - last_signal).seconds < COOLDOWN_MINUTES * 60:
            time.sleep(30)
            continue

        try:
            nifty = get_nifty_ltp()
            expiry = get_nearest_valid_expiry()
            chain = fetch_option_chain(expiry)


            if not chain:
                print(f"[{now.strftime('%H:%M:%S')}] Option chain empty for expiry {expiry}. Retrying...")
                time.sleep(30)
                continue

            atm = min(chain, key=lambda x: abs(x["strike_price"] - nifty))


            legs = []

            if atm.get("call_options"):
                legs.append((atm["call_options"], "CALL"))

            if atm.get("put_options"):
                legs.append((atm["put_options"], "PUT"))

            if not legs:
                print("ATM strike has no tradable options. Skipping.")
                time.sleep(30)
                continue

            for opt_type, label in legs:

                feats = build_feature_vector(opt_type)
                X = np.array([feats[f] for f in feature_names]).reshape(1, -1)
                Xs = scaler.transform(X)
                prob = model.predict_proba(Xs)[0][1]

                signal = generate_signal(strategy, prob, opt_type["option_greeks"])

                print(
                    f"[{now.strftime('%H:%M:%S')}] {signal} {label} | "
                    f"LTP={opt_type['market_data']['ltp']} | "
                    f"Prob={prob:.3f}"
                )

                if signal != "NO TRADE":
                    trades_today += 1
                    last_signal = now

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(30)

# ============================
# ENTRY
# ============================

if __name__ == "__main__":
    main_loop()
