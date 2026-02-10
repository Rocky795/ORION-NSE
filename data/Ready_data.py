import pandas as pd
import numpy as np

# 1. LOAD DATA
file_path = "Nifty_Hybrid_Final.csv"
print(f"Loading {file_path}...")
df = pd.read_csv(file_path, parse_dates=['ts'])
df.sort_values('ts', inplace=True) # Ensure Ascending Order (Oldest -> Newest)
df.set_index('ts', inplace=True)

# 2. FEATURE ENGINEERING (The "Hybrid" Logic)

# A. Trend Indicators (Price Action)
df['ema_20'] = df['c'].ewm(span=20, adjust=False).mean()
df['dist_from_ema'] = (df['c'] - df['ema_20']) / df['ema_20'] # Percentage distance

# B. Momentum (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['c'])

# C. Volatility (Bollinger Bands + VIX)
df['std_20'] = df['c'].rolling(20).std()
df['bb_upper'] = df['ema_20'] + (2 * df['std_20'])
df['bb_lower'] = df['ema_20'] - (2 * df['std_20'])
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ema_20']

# D. Hybrid Sentiment (The "Secret Sauce")
# OI Slope: Is "Big Money" entering or leaving?
df['oi_ma'] = df['oi'].rolling(5).mean()
df['oi_slope'] = (df['oi'] - df['oi_ma']) / df['oi_ma']

# Price-OI Divergence: Price Up + OI Up (Bullish) vs Price Up + OI Down (Weak)
df['price_change'] = df['c'].pct_change()
df['sentiment_score'] = np.where((df['price_change'] > 0) & (df['oi_change'] > 0), 1, 
                        np.where((df['price_change'] < 0) & (df['oi_change'] > 0), -1, 0))

# 3. CREATE TARGETS (Labelling for ML)
# Rule: Buy (1) if price rises 0.25% in next 4 candles (1 hour) WITHOUT dropping 0.15%
TARGET_PCT = 0.0025  # 0.25% Target (approx 60 points on Nifty 24k)
STOP_PCT = 0.0015    # 0.15% Stop Loss (approx 35 points)
LOOK_AHEAD = 4       # 4 candles = 1 Hour

indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOK_AHEAD)
df['future_high'] = df['h'].rolling(window=indexer).max()
df['future_low'] = df['l'].rolling(window=indexer).min()

# Vectorized Target Logic
df['target'] = 0
long_condition = (df['future_high'] > df['c'] * (1 + TARGET_PCT)) & \
                 (df['future_low'] > df['c'] * (1 - STOP_PCT))
df.loc[long_condition, 'target'] = 1

# 4. CLEANUP & SAVE
# Drop rows with NaNs (first 20 rows due to EMA/RSI, last 4 rows due to Target)
df_clean = df.dropna()

output_file = "Nifty_ML_Ready.csv"
df_clean.to_csv(output_file)

print(f"\nSUCCESS: Pipeline Complete!")
print(f"Features Created: RSI, EMA, Bollinger Bands, OI_Slope, Sentiment_Score")
print(f"Target Definition: +{TARGET_PCT*100}% gain within {LOOK_AHEAD} candles")
print(f"Saved {len(df_clean)} training rows to: {output_file}")
print(df_clean[['c', 'rsi', 'oi_slope', 'vix', 'target']].tail())