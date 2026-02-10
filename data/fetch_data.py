import upstox_client
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")
JSON_FILE_PATH = "NSE.json"  # Ensure this file is in the same folder

def get_keys_from_json():
    print(f"Step 1: Reading keys from {JSON_FILE_PATH}...")
    
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 1. Find NIFTY Future (Nearest Expiry)
    # Filter for NSE_FO segment, NIFTY symbol, FUT type
    mask = (df['segment'] == 'NSE_FO') & (df['name'] == 'NIFTY') & (df['instrument_type'] == 'FUT')
    nifty_futs = df[mask].copy()
    
    # Sort by expiry (expiry is in milliseconds)
    nifty_futs = nifty_futs.sort_values('expiry')
    
    # Get the first active one (Nearest Expiry)
    target_fut = nifty_futs.iloc[0]
    
    print(f"✅ Found Active Future: {target_fut['trading_symbol']}")
    print(f"   Key: {target_fut['instrument_key']}")
    
    return {
        'FUT': target_fut['instrument_key'],
        'INDEX': 'NSE_INDEX|Nifty 50',
        'VIX': 'NSE_INDEX|India VIX'
    }

def fetch_data():
    keys = get_keys_from_json()
    if not keys:
        return

    print("\nStep 2: Fetching Hybrid Data Streams...")
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_v3 = upstox_client.HistoryV3Api(upstox_client.ApiClient(configuration))
    
    # Dates: Last 30 days (Recent data for Feb 2026)
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')
    
    try:
        # 1. Fetch Futures (OI Source)
        print(f"   > Fetching Futures OI ({keys['FUT']})...")
        fut_res = api_v3.get_historical_candle_data1(keys['FUT'], "minutes", "15", to_date, from_date)
        fut_df = pd.DataFrame(fut_res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
        fut_df['ts'] = pd.to_datetime(fut_df['ts'])
        fut_df.set_index('ts', inplace=True)
        
        # 2. Fetch Index (Price Source)
        print(f"   > Fetching Index Price ({keys['INDEX']})...")
        idx_res = api_v3.get_historical_candle_data1(keys['INDEX'], "minutes", "15", to_date, from_date)
        idx_df = pd.DataFrame(idx_res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
        idx_df['ts'] = pd.to_datetime(idx_df['ts'])
        idx_df.set_index('ts', inplace=True)
        
        # 3. Fetch VIX (Volatility Source)
        print(f"   > Fetching India VIX ({keys['VIX']})...")
        vix_res = api_v3.get_historical_candle_data1(keys['VIX'], "minutes", "15", to_date, from_date)
        vix_df = pd.DataFrame(vix_res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
        vix_df['ts'] = pd.to_datetime(vix_df['ts'])
        vix_df.set_index('ts', inplace=True)

        # 4. Merge
        print("Step 3: Merging into Hybrid Dataset...")
        # Inner join ensures we only keep rows where all 3 exist
        hybrid_df = idx_df[['o', 'h', 'l', 'c']].join(
            fut_df[['oi']], how='inner'
        ).join(
            vix_df[['c']].rename(columns={'c': 'vix'}), how='inner'
        )
        
        # Feature Engineering
        hybrid_df['oi_change'] = hybrid_df['oi'].diff()
        
        filename = "Nifty_Hybrid_Final.csv"
        hybrid_df.to_csv(filename)
        print(f"\n✅ SUCCESS! Dataset Saved: {filename}")
        print(hybrid_df.tail())
        
    except Exception as e:
        print(f"❌ API Fetch Error: {e}")

if __name__ == "__main__":
    fetch_data()