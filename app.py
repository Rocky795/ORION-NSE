import time
import pandas as pd
import numpy as np
import joblib
import upstox_client
import os
import json
from datetime import datetime, timedelta
from Executions.Logic import OptionExecutionEngine # Imports your Layer-2

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")
MODEL_PATH = "models/nifty_hybrid_model.pkl" # Point to your saved model
JSON_FILE_PATH = "data/NSE.json"

# FLAGS
PAPER_TRADING = True # Set to False only when you are ready to lose real money

class LiveTrader:
    def __init__(self):
        print("Initializing Orion Live Trader...")
        
        # 1. Load Model
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("✅ Layer-1 Model Loaded.")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        # 2. Load Logic Engine
        self.engine = OptionExecutionEngine()
        print("✅ Layer-2 Logic Engine Loaded.")
        
        # 3. Setup API
        conf = upstox_client.Configuration()
        conf.access_token = UPSTOX_ACCESS_TOKEN
        self.api = upstox_client.HistoryV3Api(upstox_client.ApiClient(conf))
        
        # 4. Load Keys
        self.keys = self._get_keys()

    def _get_keys(self):
        # (Simplified version of your fetcher to get current keys)
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Get Active Nifty Future
        mask = (df['segment'] == 'NSE_FO') & (df['name'] == 'NIFTY') & (df['instrument_type'] == 'FUT')
        fut = df[mask].sort_values('expiry').iloc[0]
        
        return {
            'FUT': fut['instrument_key'],
            'INDEX': 'NSE_INDEX|Nifty 50',
            'VIX': 'NSE_INDEX|India VIX',
            'EXPIRY': fut['expiry'] # Keep expiry for Logic context
        }

    def fetch_live_features(self):
        """Fetches last 50 candles to calculate live RSI, EMA, etc."""
        print("   Fetching live market data...")
        end_str = datetime.now().strftime('%Y-%m-%d')
        start_str = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Fetch 3 streams
        # Note: In production, handle API failures/retries here
        fut_res = self.api.get_historical_candle_data1(self.keys['FUT'], "minutes", "15", end_str, start_str)
        idx_res = self.api.get_historical_candle_data1(self.keys['INDEX'], "minutes", "15", end_str, start_str)
        vix_res = self.api.get_historical_candle_data1(self.keys['VIX'], "minutes", "15", end_str, start_str)
        
        # Convert to DF
        def to_df(res):
            df = pd.DataFrame(res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            return df
        
        fut_df = to_df(fut_res)
        idx_df = to_df(idx_res)
        vix_df = to_df(vix_res)
        
        # Merge (Inner Join)
        df = idx_df[['o','h','l','c']].join(fut_df[['oi']], how='inner').join(vix_df[['c']].rename(columns={'c':'vix'}), how='inner')
        df.sort_index(inplace=True)
        
        # --- LIVE FEATURE ENGINEERING (Identical to Training) ---
        # 1. Trend
        df['ema_20'] = df['c'].ewm(span=20, adjust=False).mean()
        df['dist_from_ema'] = (df['c'] - df['ema_20']) / df['ema_20']
        
        # 2. RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. Volatility
        df['std_20'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['ema_20'] + (2*df['std_20'])
        df['bb_lower'] = df['ema_20'] - (2*df['std_20'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ema_20']
        
        # 4. Sentiment
        df['oi_ma'] = df['oi'].rolling(5).mean()
        df['oi_slope'] = (df['oi'] - df['oi_ma']) / df['oi_ma']
        
        df['price_change'] = df['c'].pct_change()
        df['oi_change'] = df['oi'].diff()
        df['sentiment_score'] = np.where((df['price_change'] > 0) & (df['oi_change'] > 0), 1, 
                                np.where((df['price_change'] < 0) & (df['oi_change'] > 0), -1, 0))
        
        # Extract LAST row (The one just closed)
        latest = df.iloc[-1]
        
        # Feature Vector (Must match training order)
        features = ['rsi', 'dist_from_ema', 'bb_width', 'oi_slope', 'sentiment_score', 'vix']
        X_live = pd.DataFrame([latest[features]])
        
        return X_live, latest['c'] # Return features and current Spot Price

    def run_cycle(self):
        """Runs one decision cycle."""
        print(f"\n--- CYCLE START: {datetime.now().strftime('%H:%M:%S')} ---")
        
        try:
            # 1. Layer-1: Prediction
            X, spot_price = self.fetch_live_features()
            prob = self.model.predict_proba(X)[0][1]
            direction = "BULLISH" # Model is binary buy/no-buy. If prob high -> Bullish. 
            # Note: If your model was trained only on Longs, low prob = Neutral/No Trade, not necessarily Bearish.
            
            print(f"   Spot: {spot_price:.2f}")
            print(f"   L1 Prediction: Prob Buy = {prob:.4f}")
            
            # 2. Prepare Layer-1 Output for Layer-2
            l1_output = {
                "direction": "BULLISH" if prob > 0.5 else "NEUTRAL", # Simplified for your Binary Model
                "probability": prob
            }
            
            # 3. Layer-2: Logic
            # Construct Market Context (Mocking IV history for now, assuming fetchable)
            market_context = {
                "spot_price": spot_price,
                "expiry_date": pd.to_datetime(self.keys['EXPIRY'], unit='ms').strftime('%Y-%m-%d'),
                "current_iv": X.iloc[0]['vix'], # Approx proxy, or fetch real Option IV
                "iv_history": [12, 13, 11, 14, 12] # Ideally fetch last 5 days VIX
            }
            
            decision = self.engine.execute_logic(l1_output, market_context)
            
            # 4. Layer-3: Execution
            print(f"   L2 Decision: {decision['trade_decision']}")
            if decision['trade_decision'] != "NO_TRADE":
                self.execute_trade(decision)
            else:
                print(f"   Reason: {decision['reason']}")
                
        except Exception as e:
            print(f"❌ Cycle Failed: {e}")

    def execute_trade(self, decision):
        symbol = f"NIFTY {decision['strike']} {decision['option_type']}"
        print(f"\n🚀 EXECUTION TRIGGERED ({'PAPER' if PAPER_TRADING else 'LIVE'})")
        print(f"   Contract: {symbol}")
        print(f"   Action:   {decision['trade_decision']}")
        print(f"   Reason:   {decision['reason']}")
        
        if not PAPER_TRADING:
            # self.api.place_order(...) 
            pass

if __name__ == "__main__":
    trader = LiveTrader()
    
    # Run immediately once to test
    trader.run_cycle()
    
    # Then loop forever (Scheduled)
    print("\n✅ System Online. Waiting for 15-min candles...")
    while True:
        # Wait until the next 15th minute (:00, :15, :30, :45)
        now = datetime.now()
        next_run = now + timedelta(minutes=15 - (now.minute % 15))
        next_run = next_run.replace(second=0, microsecond=0)
        sleep_sec = (next_run - now).total_seconds()
        
        print(f"   Sleeping {int(sleep_sec)}s until {next_run.strftime('%H:%M')}...")
        time.sleep(sleep_sec)
        
        # Wake up and run
        time.sleep(5) # Wait 5s for data to arrive at API
        trader.run_cycle()