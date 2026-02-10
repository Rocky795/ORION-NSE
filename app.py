import time
import pandas as pd
import numpy as np
import joblib
import upstox_client
import os
import json
from datetime import datetime, timedelta
# Ensure py_vollib is installed: pip install py_vollib
from py_vollib.black_scholes.implied_volatility import implied_volatility 
from Executions.Logic import OptionExecutionEngine 

# --- CONFIGURATION ---
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "Mode Training", "models", "nifty_hybrid_model.pkl")
JSON_FILE_PATH = os.path.join(SCRIPT_DIR, "data", "NSE.json")

# RISK CONTROLS (Layer-3)
PAPER_TRADING = True
MAX_TRADES_PER_DAY = 1  
RISK_FREE_RATE = 0.07   

class LiveTrader:
    def __init__(self):
        print("Initializing Orion Live Trader (Layer-3)...")
        
        # 1. Load Brains
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("✅ Layer-1 Model Loaded.")
        else:
            # Fallback path check
            alt_path = os.path.join(SCRIPT_DIR, "models", "nifty_hybrid_model.pkl")
            if os.path.exists(alt_path):
                self.model = joblib.load(alt_path)
                print("✅ Layer-1 Model Loaded (from alt path).")
            else:
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        self.engine = OptionExecutionEngine()
        print("✅ Layer-2 Logic Engine Loaded.")
        
        # 2. Setup API
        conf = upstox_client.Configuration()
        conf.access_token = UPSTOX_ACCESS_TOKEN
        self.api = upstox_client.HistoryV3Api(upstox_client.ApiClient(conf))
        
        # 3. State
        self.keys = self._get_keys()
        self.trades_today = 0
        self.iv_history = [] 

    def _get_keys(self):
        """Reads instrument keys for Nifty Index and Futures."""
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
            'EXPIRY_DATE': fut['expiry'] 
        }

    def _get_atm_contract(self, spot_price, expiry_ms):
        """Hunts for the specific ATM Call Option Key in NSE.json"""
        atm_strike = round(spot_price / 50) * 50
        
        # Load JSON to find specific option key
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Filter: NSE_FO, NIFTY, CE, ATM Strike, Correct Expiry
        mask = (
            (df['segment'] == 'NSE_FO') & 
            (df['name'] == 'NIFTY') & 
            (df['instrument_type'] == 'CE') & 
            (df['strike_price'] == atm_strike) &
            (df['expiry'] == expiry_ms)
        )
        
        results = df[mask]
        
        if not results.empty:
            contract = results.iloc[0]
            # print(f"   🎯 Found ATM Contract: {contract['trading_symbol']}")
            return contract['instrument_key'], atm_strike
        else:
            return None, None

    def _fetch_and_calculate_iv(self, instrument_key, spot, strike, expiry_ms):
        """Fetches live option price -> Calculates Implied Volatility (IV)"""
        try:
            # Get last 1-minute candle of the option to get price
            end_str = datetime.now().strftime('%Y-%m-%d')
            start_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            res = self.api.get_historical_candle_data1(instrument_key, "minutes", "1", end_str, start_str)
            
            if not res or not res.data or not res.data.candles:
                return 0.0
                
            last_close = res.data.candles[0][4] 
            
            # Time to Expiry (Years)
            expiry_date = pd.to_datetime(expiry_ms, unit='ms')
            minutes_to_expiry = (expiry_date - datetime.now()).total_seconds() / 60
            T = minutes_to_expiry / (365 * 24 * 60)
            
            if T <= 0.001: return 0.0
            
            # Reverse Engineer IV
            iv = implied_volatility(last_close, spot, strike, T, RISK_FREE_RATE, 'c')
            return iv * 100 
            
        except Exception:
            return 0.0

    def fetch_market_snapshot(self):
        print("   Fetching live market snapshot...")
        
        # 1. Get Spot & Futures (OHLC)
        end_str = datetime.now().strftime('%Y-%m-%d')
        start_str = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Fetch Candles
        fut_res = self.api.get_historical_candle_data1(self.keys['FUT'], "minutes", "15", end_str, start_str)
        idx_res = self.api.get_historical_candle_data1(self.keys['INDEX'], "minutes", "15", end_str, start_str)
        vix_res = self.api.get_historical_candle_data1(self.keys['VIX'], "minutes", "15", end_str, start_str)
        
        # Convert to DF
        def to_df(res):
            if not res or not res.data: return pd.DataFrame()
            df = pd.DataFrame(res.data.candles, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'oi'])
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            return df
        
        fut_df = to_df(fut_res)
        idx_df = to_df(idx_res)  # <--- This is defined here now
        vix_df = to_df(vix_res)
        
        # Merge for Model Input
        df = idx_df[['o','h','l','c']].join(fut_df[['oi']], how='inner').join(vix_df[['c']].rename(columns={'c':'vix'}), how='inner')
        df.sort_index(inplace=True)
        
        # 2. Get Real IV
        spot_price = idx_df.iloc[-1]['c']
        expiry_ms = self.keys['EXPIRY_DATE']
        
        # Find ATM Option & Calc IV
        atm_key, atm_strike = self._get_atm_contract(spot_price, expiry_ms)
        
        if atm_key:
            real_iv = self._fetch_and_calculate_iv(atm_key, spot_price, atm_strike, expiry_ms)
            if real_iv == 0: real_iv = vix_df.iloc[-1]['c'] # Fallback if calc fails
        else:
            real_iv = vix_df.iloc[-1]['c'] # Fallback to VIX if lookup fails
            
        self.iv_history.append(real_iv)
        
        # --- FEATURE ENGINEERING ---
        df['ema_20'] = df['c'].ewm(span=20, adjust=False).mean()
        df['dist_from_ema'] = (df['c'] - df['ema_20']) / df['ema_20']
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['std_20'] = df['c'].rolling(20).std()
        df['bb_width'] = ((df['ema_20'] + 2*df['std_20']) - (df['ema_20'] - 2*df['std_20'])) / df['ema_20']
        
        df['oi_ma'] = df['oi'].rolling(5).mean()
        df['oi_slope'] = (df['oi'] - df['oi_ma']) / df['oi_ma']
        
        df['price_change'] = df['c'].pct_change()
        df['oi_change'] = df['oi'].diff()
        df['sentiment_score'] = np.where((df['price_change'] > 0) & (df['oi_change'] > 0), 1, 
                                np.where((df['price_change'] < 0) & (df['oi_change'] > 0), -1, 0))
        
        latest = df.iloc[-1]
        features = ['rsi', 'dist_from_ema', 'bb_width', 'oi_slope', 'sentiment_score', 'vix']
        
        return pd.DataFrame([latest[features]]), latest['c'], real_iv

    def run_cycle(self):
        print(f"\n--- CYCLE START: {datetime.now().strftime('%H:%M:%S')} ---")
        
        if self.trades_today >= MAX_TRADES_PER_DAY:
            print("🛑 Max trades reached. System Halted.")
            return

        try:
            # 1. Layer-1: Prediction
            X, spot_price, current_iv = self.fetch_market_snapshot()
            prob = self.model.predict_proba(X)[0][1]
            
            print(f"   Spot: {spot_price:.2f} | Real IV: {current_iv:.2f}%")
            print(f"   L1 Prediction: Prob Buy = {prob:.4f}")
            
            l1_output = {"direction": "BULLISH" if prob > 0.5 else "NEUTRAL", "probability": prob}
            
            # 2. Layer-2: Logic
            market_context = {
                "spot_price": spot_price,
                "expiry_date": pd.to_datetime(self.keys['EXPIRY_DATE'], unit='ms').strftime('%Y-%m-%d'),
                "current_iv": current_iv,
                "iv_history": self.iv_history[-10:] if self.iv_history else [current_iv]
            }
            
            decision = self.engine.execute_logic(l1_output, market_context)
            
            # 3. Layer-3: Execution
            print(f"   L2 Decision: {decision['decision']}")
            
            if decision['decision'] == "EXECUTE":
                self.execute_complex_order(decision)
                self.trades_today += 1
            else:
                print(f"   Reason: {decision['reason']}")
                
        except Exception as e:
            print(f"❌ Cycle Failed: {e}")

    def execute_complex_order(self, decision):
        print(f"\n🚀 EXECUTION TRIGGERED ({'PAPER' if PAPER_TRADING else 'LIVE'})")
        print(f"   Strategy: {decision['strategy']}")
        print(f"   Legs:")
        for leg in decision['legs']:
            print(f"    - {leg['side']} {leg['type']} {leg['strike']}")
        print("   Status: Orders Placed (Simulated)")

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run_cycle()
    
    print("\n✅ System Online.")
    while True:
        now = datetime.now()
        next_run = now + timedelta(minutes=15 - (now.minute % 15))
        next_run = next_run.replace(second=0, microsecond=0)
        sleep_sec = (next_run - now).total_seconds()
        
        print(f"   Sleeping {int(sleep_sec)}s until {next_run.strftime('%H:%M')}...")
        time.sleep(sleep_sec)
        time.sleep(5) 
        trader.run_cycle()