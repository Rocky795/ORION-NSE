import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
RAW_FILE = os.path.join(DATA_DIR, "Nifty_Hybrid_60days.csv")

# STRATEGY SETTINGS (The "One Change" + Sequential Logic)
LOOK_AHEAD = 8         # 8 Candles (2 Hours)
TP_PCT = 0.0025        # 0.25% Target
SL_PCT = 0.0015        # 0.15% Stop Loss
PROB_THRESHOLD = 0.65  # Confidence Threshold

# 1. LOAD DATA
print(f"Loading Raw Data: {RAW_FILE}...")
try:
    df = pd.read_csv(RAW_FILE, parse_dates=['ts'], index_col='ts')
    df.sort_index(inplace=True)
except FileNotFoundError:
    print("❌ Raw data not found. Run your data fetcher again.")
    exit()

print("Calculating Features...")
# Technicals
df['ema_20'] = df['c'].ewm(span=20, adjust=False).mean()
df['dist_from_ema'] = (df['c'] - df['ema_20']) / df['ema_20']
df['std_20'] = df['c'].rolling(20).std()
df['bb_width'] = ((df['ema_20'] + 2*df['std_20']) - (df['ema_20'] - 2*df['std_20'])) / df['ema_20']

# RSI
delta = df['c'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Hybrid Sentiment
df['oi_slope'] = (df['oi'] - df['oi'].rolling(5).mean()) / df['oi'].rolling(5).mean()
df['price_change'] = df['c'].pct_change()
df['oi_change'] = df['oi'].diff()
df['sentiment_score'] = np.where((df['price_change'] > 0) & (df['oi_change'] > 0), 1, 
                        np.where((df['price_change'] < 0) & (df['oi_change'] > 0), -1, 0))

# --- THE FIX: SEQUENTIAL LABELING (First-Hit Logic) ---
print(f"Applying Sequential Labeling (Horizon: {LOOK_AHEAD})...")

# Convert to numpy for speed
closes = df['c'].values
highs = df['h'].values
lows = df['l'].values
n = len(df)
targets = np.zeros(n)

# Iterate row by row (The only way to be 100% accurate on path)
for i in range(n - LOOK_AHEAD):
    entry_price = closes[i]
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)
    
    outcome = 0 # Default is 0 (Time Stop / Flat)
    
    # Check the next N candles sequentially
    for j in range(1, LOOK_AHEAD + 1):
        future_high = highs[i + j]
        future_low = lows[i + j]
        
        # CRITICAL: Check SL first (Conservative Assumption)
        # If low hits SL, it's a loss, even if high hits TP in same candle.
        if future_low <= sl_price:
            outcome = 0
            break # Trade over, stopped out
            
        if future_high >= tp_price:
            outcome = 1
            break # Trade over, take profit
            
    targets[i] = outcome

df['target'] = targets

# Cleanup
df.dropna(inplace=True)
features = ['rsi', 'dist_from_ema', 'bb_width', 'oi_slope', 'sentiment_score', 'vix']
X = df[features]
y = df['target']

# --- DIAGNOSTIC 1: REALITY CHECK ---
print(f"\n--- DIAGNOSTIC 1: Class Balance (Sequential Logic) ---")
buy_signal_pct = y.mean() * 100
print(f"True Win Rate (Buy Signals): {buy_signal_pct:.2f}% of dataset")

if buy_signal_pct < 10:
    print("⚠️ WARNING: Target is extremely rare. TP might be too far or SL too tight.")
    print("   -> Action: Increase Horizon or Tighten TP.")

# --- DIAGNOSTIC 2: PROBABILITY CHECK ---
print(f"\n--- DIAGNOSTIC 2: Model Confidence Check ---")
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Check if we have enough positive cases to train
    if y_train.sum() < 5:
        print("   [Skip Fold] Not enough positive samples to train.")
        continue

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    weight = neg / pos if pos > 0 else 1
    
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        scale_pos_weight=weight, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"Split: Mean Prob={probs.mean():.3f} | Max Prob={probs.max():.3f}")
    
    preds = (probs >= PROB_THRESHOLD).astype(int)
    precision = precision_score(y_test, preds, zero_division=0)
    print(f"   Trades: {preds.sum()} | Precision: {precision:.2f}")

print("\n INTERPRETATION:")
print("1. If 'Buy Signals' dropped significantly (e.g., from 40% to 15%), the old logic was lying to you.")
print("2. If 'Max Prob' is now < 0.60, the model effectively says 'I see no safe trades'.")