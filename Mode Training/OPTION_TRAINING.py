import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score
from sklearn.utils import class_weight

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Directory Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
DATA_FILE = os.path.join(DATA_DIR, 'nifty_data.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Helper Classes ---
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types to fix JSON serialization errors """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class FeatureEngineer:
    """Handles feature creation and normalization for Nifty Options Strategy."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Generates technical indicators."""
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
        
        if is_training:
            df.dropna(inplace=True)
            
        self.feature_cols = [
            'log_ret_1', 'log_ret_3', 'log_ret_5', 'log_ret_15',
            'atr_14', 'std_dev_20', 
            'sma_slope_20', 'sma_slope_50', 
            'rel_vol', 'hour', 'day_of_week'
        ]
        return df

    def scale_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if is_training:
            df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        else:
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

# --- Mock Data Generator ---
def generate_dummy_data(filepath: str, n_rows: int = 10000) -> None:
    logger.warning(f"Data file not found at {filepath}. Generating dummy data...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_rows, freq='5min')
    base_price = 22000.0
    returns = np.random.normal(0, 0.001, n_rows)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Datetime': dates,
        'Open': price_path,
        'High': price_path * (1 + np.random.rand(n_rows) * 0.001),
        'Low': price_path * (1 - np.random.rand(n_rows) * 0.001),
        'Close': price_path * (1 + np.random.normal(0, 0.0005, n_rows)),
        'Volume': np.random.randint(5000, 150000, n_rows)
    })
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    data.to_csv(filepath, index=False)
    logger.info("Dummy data generated successfully.")

# --- Label Generation ---
def generate_labels(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.0015) -> pd.DataFrame:
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
    
    conditions = [
        (df['future_return'] > threshold),
        (df['future_return'] < -threshold)
    ]
    choices = [1, 2]
    df['target'] = np.select(conditions, choices, default=0)
    df.dropna(subset=['future_return'], inplace=True)
    logger.info(f"Label Distribution:\n{df['target'].value_counts()}")
    return df

# --- Main Execution Flow ---
def main():
    # --- Step 1: Data Ingestion ---
    logger.info("Step 1: Checking Data Source...")
    
        
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Cleaning
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is not None:
        df['Datetime'] = df['Datetime'].dt.tz_convert(None)
    df.ffill(inplace=True) 
    
    # --- Step 2: Feature Engineering ---
    logger.info("Step 2: Engineering Features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df, is_training=True)
    
    # --- Step 3: Label Generation ---
    logger.info("Step 3: Generating Labels (Truth)...")
    df_labeled = generate_labels(df_features, horizon=6, threshold=0.0015)
    
    # --- Step 4: Normalization ---
    logger.info("Step 4: Normalizing Data...")
    df_final = engineer.scale_data(df_labeled, is_training=True)
    
    # Prepare X and y
    X = df_final[engineer.feature_cols]
    y = df_final['target']

    # --- Step 5: Model Training ---
    logger.info("Step 5: Training Random Forest with TimeSeriesSplit...")
    
    # Handling Class Imbalance
    unique_classes = np.unique(y)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y
    )
    
    # Convert keys to standard int and values to standard float
    weights_dict = {
        int(k): float(v) for k, v in zip(unique_classes, class_weights)
    }
    logger.info(f"Class Weights: {weights_dict}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight=weights_dict,
        n_jobs=-1,
        random_state=42
    )

    # Validation Loop
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        
        precision = precision_score(y_val, preds, average='weighted', zero_division=0)
        logger.info(f"Fold {fold} Weighted Precision: {precision:.4f}")
        fold += 1

    # Final fit on all data for production
    logger.info("Finalizing model on full dataset...")
    clf.fit(X, y)
    
    # --- Step 6: Saving Artifacts ---
    logger.info("Step 6: Saving Model Artifacts...")
    
    # 1. Save Model
    joblib.dump(clf, os.path.join(MODELS_DIR, 'nifty_options_model.joblib'))
    
    # 2. Save Scaler
    joblib.dump(engineer.scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    # 3. Save Metadata
    metadata = {
        "features": engineer.feature_cols,
        "classes": {0: "No Trade", 1: "Buy Call", 2: "Buy Put"},
        "threshold": 0.0015,
        "horizon_candles": 6,
        "model_params": clf.get_params()
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
    
    # Use NumpyEncoder to handle any remaining numpy types in params
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
        
    logger.info(f"Success! Artifacts saved in {MODELS_DIR}")

if __name__ == "__main__":
    main()