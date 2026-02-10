import upstox_client
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")

# CRITICAL FIX: Get the folder where THIS script lives (i.e., the 'data' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input File (Force it to look in data folder)
JSON_FILE_PATH = os.path.join(SCRIPT_DIR, "NSE.json")

# Output Files (Force them to save in data folder)
RAW_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Nifty_Hybrid_60days.csv")
FINAL_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "Nifty_ML_Ready.csv")

# --- PART 1: DATA COLLECTION FUNCTIONS ---

def get_keys_from_json():
    """Reads the current active keys from your JSON file."""
    print(f"Step 1: Reading keys from {JSON_FILE_PATH}...")
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        mask = (df['segment'] == 'NSE_FO') & (df['name'] == 'NIFTY') & (df['instrument_type'] == 'FUT')
        nifty_futs = df[mask].copy()
        
        if 'expiry' in nifty_futs.columns:
            nifty_futs = nifty_futs.sort_values('expiry')
        
        target_fut = nifty_futs.iloc[0]
        
        print(f"✅ Active Future: {target_fut['trading_symbol']} ({target_fut['instrument_key']})")
        
        return {
            'FUT': target_fut['instrument_key'],
            'INDEX': 'NSE_INDEX|Nifty 50',
            'VIX': 'NSE_INDEX|India VIX'
        }
    except Exception as e:
        print(f"❌ Error reading JSON keys: {e}")
        return None

def fetch_chunked_data(api_instance, instrument_key, days_to_fetch=60, interval="15"):
    """Fetches data in 30-day chunks to bypass API limits."""
    print(f"   Fetching {days_to_fetch} days for {instrument_key}...")
    
    all_dfs = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    
    current_to = end_date
    while current_to > start_date:
        current_from = max(current_to - timedelta(days=25), start_date)
        
        to_str = current_to.strftime('%Y-%m-%d')
        from_str = current_from.strftime('%Y-%m-%d')
        
        try:
            res = api_instance.get_historical_candle_data1(instrument_key, "minutes", interval, to_str, from_str)
            if res.status == 'success' and res.data.candles:
                chunk_df = pd.DataFrame(res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
                chunk_df['ts'] = pd.to_datetime(chunk_df['ts'])
                chunk_df.set_index('ts', inplace=True)
                all_dfs.append(chunk_df)
            else:
                print(f"      No data for {from_str} to {to_str}")
        except Exception as e:
            print(f"      Error fetching chunk: {e}")
            
        current_to = current_from - timedelta(days=1)
        time.sleep(0.5)

    if all_dfs:
        full_df = pd.concat(all_dfs)
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        return full_df.sort_index()
    return None

# --- PART 2: FEATURE ENGINEERING FUNCTIONS ---

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def apply_feature_engineering(df):
    """Applies technical indicators and creates target variables."""
    print("\nStep 4: Applying Feature Engineering...")
    
    # 1. Trend Indicators
    df['ema_20'] = df['c'].ewm(span=20, adjust=False).mean()
    df['dist_from_ema'] = (df['c'] - df['ema_20']) / df['ema_20']

    # 2. Momentum
    df['rsi'] = calculate_rsi(df['c'])

    # 3. Volatility
    df['std_20'] = df['c'].rolling(20).std()
    df['bb_upper'] = df['ema_20'] + (2 * df['std_20'])
    df['bb_lower'] = df['ema_20'] - (2 * df['std_20'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ema_20']

    # 4. Hybrid Sentiment
    df['oi_ma'] = df['oi'].rolling(5).mean()
    df['oi_slope'] = (df['oi'] - df['oi_ma']) / df['oi_ma']

    df['price_change'] = df['c'].pct_change()
    if 'oi_change' not in df.columns:
        df['oi_change'] = df['oi'].diff()
        
    df['sentiment_score'] = np.where((df['price_change'] > 0) & (df['oi_change'] > 0), 1, 
                            np.where((df['price_change'] < 0) & (df['oi_change'] > 0), -1, 0))

    # 5. CREATE TARGETS
    TARGET_PCT = 0.0025
    STOP_PCT = 0.0015
    LOOK_AHEAD = 4

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOK_AHEAD)
    df['future_high'] = df['h'].rolling(window=indexer).max()
    df['future_low'] = df['l'].rolling(window=indexer).min()

    df['target'] = 0
    long_condition = (df['future_high'] > df['c'] * (1 + TARGET_PCT)) & \
                     (df['future_low'] > df['c'] * (1 - STOP_PCT))
    df.loc[long_condition, 'target'] = 1
    
    return df

# --- MAIN PIPELINE ---

def main():
    keys = get_keys_from_json()
    if not keys:
        print("Exiting: Could not find keys.")
        return
        
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_v3 = upstox_client.HistoryV3Api(upstox_client.ApiClient(configuration))
    
    DAYS = 60
    print(f"\nStep 2: Starting {DAYS}-Day Data Collection...")
    
    fut_df = fetch_chunked_data(api_v3, keys['FUT'], days_to_fetch=DAYS)
    idx_df = fetch_chunked_data(api_v3, keys['INDEX'], days_to_fetch=DAYS)
    vix_df = fetch_chunked_data(api_v3, keys['VIX'], days_to_fetch=DAYS)
    
    if fut_df is not None and idx_df is not None and vix_df is not None:
        print("\nStep 3: Merging Data Streams...")
        hybrid_df = idx_df[['o', 'h', 'l', 'c']].join(
            fut_df[['oi']], how='inner'
        ).join(
            vix_df[['c']].rename(columns={'c': 'vix'}), how='inner'
        )
        
        hybrid_df['oi_change'] = hybrid_df['oi'].diff()
        
        # SAVE RAW DATA (Using enforced path)
        hybrid_df.to_csv(RAW_OUTPUT_FILE)
        print(f"✅ Saved Raw Data: {RAW_OUTPUT_FILE} ({len(hybrid_df)} rows)")
        
        # PROCESS PIPELINE
        processed_df = apply_feature_engineering(hybrid_df)
        df_clean = processed_df.dropna()
        
        # SAVE FINAL DATA (Using enforced path)
        df_clean.to_csv(FINAL_OUTPUT_FILE)

        print(f"\nSUCCESS: Pipeline Complete!")
        print(f"Final Dataset Saved to: {FINAL_OUTPUT_FILE}")
        print(df_clean[['c', 'rsi', 'oi_slope', 'vix', 'target']].tail())
        
    else:
        print("\n❌ Failed to fetch complete data for all streams.")

if __name__ == "__main__":
    main()