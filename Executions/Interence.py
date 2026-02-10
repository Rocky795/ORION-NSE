import os
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime
from typing import Dict, Any, List

# --- Configuration ---
# Threshold for high-probability setup (0.60 = 60% confidence)
CONFIDENCE_THRESHOLD = 0.60 
MIN_HISTORY_REQUIRED = 52  # Need at least 52 candles for SMA-50 + diffs

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NiftyInference")

# --- Directory Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'nifty_data.csv')

# --- 1. Feature Engineering (MUST MATCH TRAINING EXACTLY) ---
class FeatureEngineer:
    """
    Replicates the exact feature generation logic from the training phase.
    """
    def __init__(self):
        self.feature_cols = [] 

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Momentum: Log Returns
        for window in [1, 3, 5, 15]:
            df[f'log_ret_{window}'] = np.log(df['Close'] / df['Close'].shift(window))
            
        # 2. Volatility
        df['atr_14'] = self._compute_atr(df, 14)
        df['std_dev_20'] = df['Close'].rolling(20).std()
        
        # 3. Trend: Slope of SMAs
        for p in [20, 50]:
            sma = df['Close'].rolling(p).mean()
            df[f'sma_slope_{p}'] = (sma - sma.shift(1)) / sma.shift(1)
            
        # 4. Relative Volume
        df['rel_vol'] = df['Volume'] / df['Volume'].rolling(50).mean()
        
        # 5. Time Features
        df['hour'] = df['Datetime'].dt.hour
        df['day_of_week'] = df['Datetime'].dt.dayofweek
        
        # Note: In inference, we DO NOT dropna() the whole df, 
        # we just need the last row to be valid.
        return df

# --- 2. Artifact Loader ---
def load_artifacts():
    """Loads model, scaler, and metadata."""
    try:
        logger.info("Loading model artifacts...")
        model = joblib.load(os.path.join(MODELS_DIR, 'nifty_options_model.joblib'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        logger.info("Artifacts loaded successfully.")
        return model, scaler, metadata
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        logger.error("Did you run the training script first?")
        exit(1)

# --- 3. Core Inference Logic ---
def get_prediction(
    df_slice: pd.DataFrame, 
    model: Any, 
    scaler: Any, 
    metadata: Dict
) -> Dict[str, Any]:
    """
    Takes a slice of market data, generates features for the LAST candle,
    and returns a trade signal.
    """
    # 1. Feature Engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df_slice)
    
    # Get the last row (current candle)
    last_row = df_features.iloc[[-1]].copy()
    
    # Check if we have enough data (NaN check)
    # If SMA-50 is NaN, we can't predict.
    if last_row[metadata['features']].isna().any().any():
        return {"signal": "WAIT_DATA", "confidence": 0.0, "reason": "Insufficient History"}

    # 2. Scale Features
    # Ensure columns are in the EXACT order as training
    X_input = last_row[metadata['features']]
    X_scaled = scaler.transform(X_input)
    
    # 3. Predict Probability
    # classes are usually [0, 1, 2] -> [No Trade, Call, Put]
    probs = model.predict_proba(X_scaled)[0] # Returns array like [0.2, 0.7, 0.1]
    
    # Map classes
    class_map = {int(k): v for k, v in metadata['classes'].items()}
    
    # 4. Apply Threshold Logic
    # Default
    signal = "NO_TRADE"
    confidence = probs[0] # Prob of No Trade
    
    # Check Call (Class 1)
    if probs[1] > CONFIDENCE_THRESHOLD:
        signal = "BUY_CALL"
        confidence = probs[1]
        
    # Check Put (Class 2)
    elif probs[2] > CONFIDENCE_THRESHOLD:
        signal = "BUY_PUT"
        confidence = probs[2]
        
    return {
        "timestamp": last_row['Datetime'].iloc[0],
        "close": last_row['Close'].iloc[0],
        "signal": signal,
        "confidence": confidence,
        "probs": probs
    }

# --- 4. Simulated Live Stream ---
def run_simulation():
    model, scaler, metadata = load_artifacts()
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found. Cannot simulate.")
        return

    logger.info("Loading market data for simulation...")
    full_df = pd.read_csv(DATA_FILE)
    full_df['Datetime'] = pd.to_datetime(full_df['Datetime'])
    
    # Convert to naive datetime to match training logic
    if full_df['Datetime'].dt.tz is not None:
        full_df['Datetime'] = full_df['Datetime'].dt.tz_convert(None)
    
    logger.info(f"Starting Simulation with {len(full_df)} candles.")
    logger.info("---------------------------------------------------------------------------------")
    logger.info(f"{'TIMESTAMP':<20} | {'PRICE':<10} | {'SIGNAL':<10} | {'CONF':<6} | {'STATUS'}")
    logger.info("---------------------------------------------------------------------------------")

    # Start simulation from the 100th candle to ensure we have enough history for SMA-50
    start_index = 100
    
    for i in range(start_index, len(full_df)):
        # Simulate "Live" feed by taking a slice of data up to current index
        # In production, this would be: df = fetch_last_200_candles()
        live_slice = full_df.iloc[i-100 : i+1].copy() # Keep last 100 rows for indicators
        
        current_candle = live_slice.iloc[-1]
        
        # TIME FILTER: Only trade between 09:15 and 15:30
        curr_time = current_candle['Datetime'].time()
        market_open = dtime(9, 15)
        market_close = dtime(15, 30)
        
        if not (market_open <= curr_time <= market_close):
             # Skip printing every single off-market line to keep logs clean
             continue

        # Generate Prediction
        result = get_prediction(live_slice, model, scaler, metadata)
        
        # Formatting Output
        ts_str = result['timestamp'].strftime("%Y-%m-%d %H:%M")
        price = f"{result['close']:.2f}"
        conf_pct = f"{result['confidence']*100:.1f}%"
        sig = result['signal']
        
        # Color coding for terminal (Optional, uses ANSI codes)
        RESET = "\033[0m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        
        status_color = RESET
        if sig == "BUY_CALL": status_color = GREEN
        elif sig == "BUY_PUT": status_color = RED
        elif sig == "WAIT_DATA": status_color = YELLOW
        
        # Log to console
        print(f"{ts_str:<20} | {price:<10} | {status_color}{sig:<10}{RESET} | {conf_pct:<6} | P(C):{result['probs'][1]:.2f} P(P):{result['probs'][2]:.2f}")
        
        # Simulation Delay
        # Reduce sleep to 0.1s for fast-forward testing, or 1.0s for "real-feel"
        time.sleep(0.5) 

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")