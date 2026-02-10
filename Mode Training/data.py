import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_nifty_history_60d():
    """
    Generates a realistic 60-day dataset for NIFTY Options
    combining yfinance spot data with synthetic Greeks.
    """
    # 1. Get Real Spot Data (The "Truth")
    import yfinance as yf
    print("Fetching last 60 days of Nifty Spot Data...")
    df = yf.download("^NSEI", period="60d", interval="15m", progress=False)
    
    if df.empty:
        print("Error: Could not fetch data.")
        return pd.DataFrame()
        
    # Flatten & Clean
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "timestamp", "Datetime": "timestamp", "Close": "close"}, inplace=True)
    
    # 2. Add Synthetic Option Metrics (The "Proxy")
    # We simulate IV and OI based on price action to make it "trainable"
    
    # IV rises when Nifty falls (Fear factor)
    df['atm_iv'] = 15 + (20000 - df['close']) / 1000 + np.random.normal(0, 1, len(df))
    df['atm_iv'] = df['atm_iv'].clip(10, 30) # Keep IV between 10 and 30
    
    # PCR tracks trend (Bullish trend = Higher PCR)
    df['pcr'] = 0.5 + (df['close'].rolling(50).rank(pct=True)) 
    
    # OI Sentiment (1 = Bullish, 0 = Bearish)
    # If price is above SMA-20, assume bullish OI flow
    df['sma_20'] = df['close'].rolling(20).mean()
    df['oi_sentiment'] = np.where(df['close'] > df['sma_20'], 1, 0)
    
    # Minutes to Expiry (Simulated weekly cycle)
    # Assuming every 400 candles (~5 days) is an expiry
    df['minutes_to_expiry'] = 1500 - (df.index % 400) * 3.75
    df['minutes_to_expiry'] = df['minutes_to_expiry'].clip(lower=0)
    
    # Drop NaN from rolling calcs
    df.dropna(inplace=True)
    
    print(f"Generated {len(df)} rows of Hybrid Training Data.")
    return df

if __name__ == "__main__":
    data = fetch_nifty_history_60d()
    print(data.head())
    # Save to CSV for your trainer
    data.to_csv("nifty_hybrid_60d.csv", index=False)
    print("Saved to nifty_hybrid_60d.csv")