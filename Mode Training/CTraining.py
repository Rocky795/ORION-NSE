import numpy as np
import pandas as pd
import xgboost as xgb
import pandas_ta as ta  # Technical Analysis library
from sklearn.metrics import precision_score, classification_report
from datetime import timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. MOCK DATA GENERATOR (Simulating the Hybrid Feed)
# ==========================================
def generate_hybrid_data(days=200):
    """
    Generates synthetic Nifty 15-min data AND Option Metrics (IV, PCR, OI).
    Real-world equivalent: Merging NSE Index data with Option Chain snapshots.
    """
    # 15-minute frequency, 25 candles per day (approx 6.25 hours)
    n_rows = days * 25
    date_rng = pd.date_range(start='2023-01-01', periods=n_rows, freq='15T')
    
    # 1. Index Price Simulation (Random Walk with Drift)
    price = 20000 + np.cumsum(np.random.normal(0, 20, n_rows))
    
    # 2. Option Metrics Simulation
    # IV tends to rise as markets fall (fear), but here we simulate random regimes
    iv = np.random.normal(15, 2, n_rows) 
    
    # PCR (Put Call Ratio): Higher often bullish (oversold), Lower bearish
    pcr = np.random.uniform(0.5, 1.5, n_rows)
    
    # OI Change: Rising OI + Rising Price = Long Build Up
    oi_change_atm = np.random.normal(0, 50000, n_rows)
    
    # 3. Time to Expiry (Weekly Cycle Simulation)
    # Reset countdown every 100 candles (approx 4 days)
    minutes_to_expiry = [((100 - (i % 100)) * 15) for i in range(n_rows)]

    df = pd.DataFrame({
        'timestamp': date_rng,
        'open': price + np.random.normal(0, 5, n_rows),
        'high': price + np.random.normal(10, 5, n_rows),
        'low': price - np.random.normal(10, 5, n_rows),
        'close': price,
        'volume': np.random.randint(1000, 100000, n_rows),
        'atm_iv': iv,
        'pcr': pcr,
        'oi_change_atm': oi_change_atm,
        'minutes_to_expiry': minutes_to_expiry
    })
    
    df.set_index('timestamp', inplace=True)
    return df

# ==========================================
# 2. FEATURE ENGINEERING (The "Volatility Blindness" Cure)
# ==========================================
def engineer_features(df):
    """
    Creates the Hybrid Feature Vector.
    """
    df = df.copy()
    
    # --- A. Technicals (Index) ---
    # RSI 14
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # EMAs for Trend
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    
    # Crossover Signal (Trend Confirm)
    df['trend_up'] = np.where(df['ema_50'] > df['ema_200'], 1, 0)
    
    # Distance from EMA (Mean Reversion / Extension check)
    df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['ema_20']

    # --- B. Option Dynamics (The Hybrid Mix) ---
    # IV Regime: Is volatility expanding?
    df['iv_slope'] = df['atm_iv'].diff(3) # Change in IV over last 45 mins
    
    # PCR Signal
    df['pcr_signal'] = np.where(df['pcr'] > 1.2, 1, 0) # Bullish support
    
    # Smart Money Flow (OI * Volume interaction)
    # If Price Up AND OI Up -> Long Buildup (Strong Buy)
    df['oi_sentiment'] = np.where((df['close'].diff() > 0) & (df['oi_change_atm'] > 0), 1, 0)
    
    # --- C. Time Decay Protection ---
    # Avoid buying options when expiry is too close (Theta risk)
    df['safe_expiry_window'] = np.where(df['minutes_to_expiry'] > 120, 1, 0)

    # Drop NaN from indicator warm-up
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. LABELING (The "Truth")
# ==========================================
def create_targets(df, profit_threshold=0.002, stop_loss=0.001):
    """
    Logic: CE BUY (1) if:
    1. Next 2 candles (30 mins) cumulative return > profit_threshold (0.2%)
    2. We do NOT hit the stop loss in that timeframe.
    """
    df = df.copy()
    
    # Look forward 2 periods (30 mins)
    future_high = df['high'].shift(-2).rolling(2).max()
    future_low = df['low'].shift(-2).rolling(2).min()
    current_close = df['close']
    
    # Reward condition: Did we go up enough?
    upside_move = (future_high - current_close) / current_close
    
    # Risk condition: Did we crash first?
    downside_risk = (current_close - future_low) / current_close
    
    # Target: 1 if Reward > Threshold AND Risk < Stop Loss
    df['target'] = np.where(
        (upside_move > profit_threshold) & (downside_risk < stop_loss), 
        1, 0
    )
    
    # Shift labels back to align with features (Predicting the future)
    # Note: The logic above already looks forward, but we need to ensure 
    # the target at time T represents the outcome at T+2.
    # The 'shift(-2)' implies we are looking at future data.
    # For a trading model, if row T has features, Target T describes outcome T+1, T+2.
    # The calculation `shift(-2)` puts the future value on the current row. 
    # So no extra shift is needed here.
    
    return df.dropna()

# ==========================================
# 4. WALK-FORWARD TRAINING ENGINE
# ==========================================
class WalkForwardTrainer:
    def __init__(self, df, window_days=60, step_days=7):
        self.df = df
        self.window_size = window_days * 25 # 25 candles/day
        self.step_size = step_days * 25
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        self.results = []
        
    def train(self):
        """
        Sliding Window:
        Train: [Start : Start + Window]
        Test:  [Start + Window : Start + Window + Step]
        Slide: Start += Step
        """
        total_samples = len(self.df)
        current_start = 0
        
        feature_cols = [
            'rsi_14', 'dist_ema_20', 'trend_up', 
            'atm_iv', 'iv_slope', 'pcr', 'oi_sentiment', 
            'minutes_to_expiry'
        ]
        
        print(f"Starting Walk-Forward Training ({len(feature_cols)} features)...")
        print(f"{'Train Period':<20} | {'Test Period':<20} | {'Precision':<10} | {'Trades'}")
        print("-" * 75)
        
        while current_start + self.window_size + self.step_size < total_samples:
            # 1. Define Slices
            train_end = current_start + self.window_size
            test_end = train_end + self.step_size
            
            train_data = self.df.iloc[current_start : train_end]
            test_data = self.df.iloc[train_end : test_end]
            
            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_test = test_data[feature_cols]
            y_test = test_data['target']
            
            # 2. Train Model
            # Scale_pos_weight helps with class imbalance (Buy signals are rare)
            ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
            self.model.set_params(scale_pos_weight=ratio)
            
            self.model.fit(X_train, y_train)
            
            # 3. Predict & Evaluate
            preds = self.model.predict(X_test)
            precision = precision_score(y_test, preds, zero_division=0)
            
            # Store results
            self.results.append({
                'start_date': train_data.index[0].date(),
                'test_date': test_data.index[0].date(),
                'precision': precision,
                'trades': sum(preds)
            })
            
            print(f"{str(train_data.index[0].date()):<20} | {str(test_data.index[0].date()):<20} | {precision:.4f}     | {sum(preds)}")
            
            # Slide Window
            current_start += self.step_size

    def get_performance_summary(self):
        res_df = pd.DataFrame(self.results)
        avg_precision = res_df['precision'].mean()
        print("\n=== Performance Summary ===")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Total Weeks Tested: {len(res_df)}")
        return res_df

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Generate Data
    print("Generating Hybrid Market Data...")
    raw_df = generate_hybrid_data(days=365)
    
    # 2. Engineer Features
    print("Engineering Features (IV, OI, Greeks)...")
    processed_df = engineer_features(raw_df)
    
    # 3. Create Labels (Risk:Reward)
    labeled_df = create_targets(processed_df)
    
    # 4. Run Walk-Forward Validation
    trainer = WalkForwardTrainer(labeled_df, window_days=60, step_days=7)
    trainer.train()
    stats = trainer.get_performance_summary()