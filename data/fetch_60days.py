import upstox_client
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")
JSON_FILE_PATH = "NSE.json"  # Uses your uploaded file

def get_keys_from_json():
    """Reads the current active keys from your JSON file."""
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Find NIFTY Future (Nearest Expiry)
    mask = (df['segment'] == 'NSE_FO') & (df['name'] == 'NIFTY') & (df['instrument_type'] == 'FUT')
    nifty_futs = df[mask].sort_values('expiry')
    target_fut = nifty_futs.iloc[0]
    
    print(f"✅ Active Future: {target_fut['trading_symbol']} ({target_fut['instrument_key']})")
    
    return {
        'FUT': target_fut['instrument_key'],
        'INDEX': 'NSE_INDEX|Nifty 50',
        'VIX': 'NSE_INDEX|India VIX'
    }

def fetch_chunked_data(api_instance, instrument_key, days_to_fetch=60, interval="15"):
    """Fetches data in 30-day chunks to bypass API limits."""
    print(f"   Fetching {days_to_fetch} days for {instrument_key}...")
    
    all_dfs = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    
    # Loop in 25-day chunks to be safe (limit is 30)
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
                # print(f"      Got chunk: {from_str} to {to_str} ({len(chunk_df)} rows)")
            else:
                print(f"      No data for {from_str} to {to_str}")
        except Exception as e:
            print(f"      Error fetching chunk: {e}")
            
        # Move back in time
        current_to = current_from - timedelta(days=1)
        time.sleep(0.5) # Rate limit protection

    if all_dfs:
        # Combine all chunks and sort
        full_df = pd.concat(all_dfs)
        full_df = full_df[~full_df.index.duplicated(keep='first')] # Remove overlaps
        return full_df.sort_index()
    return None

def main():
    keys = get_keys_from_json()
    
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_v3 = upstox_client.HistoryV3Api(upstox_client.ApiClient(configuration))
    
    # --- FETCH 60 DAYS FOR ALL 3 STREAMS ---
    DAYS = 60
    
    print("\n--- Starting 60-Day Data Collection ---")
    fut_df = fetch_chunked_data(api_v3, keys['FUT'], days_to_fetch=DAYS)
    idx_df = fetch_chunked_data(api_v3, keys['INDEX'], days_to_fetch=DAYS)
    vix_df = fetch_chunked_data(api_v3, keys['VIX'], days_to_fetch=DAYS)
    
    # --- MERGE ---
    if fut_df is not None and idx_df is not None and vix_df is not None:
        print("\nMerging Data...")
        hybrid_df = idx_df[['o', 'h', 'l', 'c']].join(
            fut_df[['oi']], how='inner'
        ).join(
            vix_df[['c']].rename(columns={'c': 'vix'}), how='inner'
        )
        
        # Calculate Hybrid Features
        hybrid_df['oi_change'] = hybrid_df['oi'].diff()
        
        filename = f"Nifty_Hybrid_{DAYS}days.csv"
        hybrid_df.to_csv(filename)
        print(f"\n✅ SUCCESS! Saved {len(hybrid_df)} rows to {filename}")
        print(f"Date Range: {hybrid_df.index.min()} to {hybrid_df.index.max()}")
    else:
        print("\n❌ Failed to fetch complete data.")

if __name__ == "__main__":
    main()