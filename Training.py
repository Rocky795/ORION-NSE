
"""
NSE Options Profitability Prediction Model
A complete end-to-end ML pipeline for predicting ATM option trade profitability

Author: Quantitative Trading System
Purpose: Train models to predict if buying ATM options will be profitable within 3 candles
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data fetching


# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

# Technical indicators
from scipy import stats
import yfinance as yf



def fetch_data(symbol='^NSEI', start_date=None, end_date=None, save_path='nifty_data.csv'):
    """
    Fetch historical OHLCV data using yfinance (stable replacement for nsepy)
    """

    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365*10)

    print(f"Fetching data from {start_date.date()} to {end_date.date()} using yfinance...")

    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval="5m",
            progress=False
        )

        if df is None or df.empty:
            raise ValueError("No data returned from yfinance")

        # Standardize column names
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        df.index.name = 'Date'
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        df.to_csv(save_path)
        print(f"Data saved to {save_path}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    except Exception as e:
        print("❌ Data fetch failed.")
        print(str(e))
        return None


def create_features(df):
    """
    Create technical indicators and features from OHLCV data
    
    WHY: Raw prices are not stationary. Technical indicators capture market dynamics
    that are predictive of future price movements.
    
    CRITICAL: All features must be calculated using ONLY past data (no future leakage)
    """
    
    print("Creating features...")
    
    df = df.copy()
    
    # 1. LOG RETURNS (different periods)
    # WHY: Returns are more stationary than prices, log returns are symmetric
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_3'] = np.log(df['Close'] / df['Close'].shift(3))
    df['log_return_5'] = np.log(df['Close'] / df['Close'].shift(5))
    
    # 2. RSI (Relative Strength Index)
    # WHY: Identifies overbought/oversold conditions which often precede reversals
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = calculate_rsi(df['Close'], period=14)
    
    # 3. VWAP (Volume Weighted Average Price) and distance from it
    # WHY: VWAP is a key institutional benchmark. Price tends to revert to VWAP
    df['vwap'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['distance_from_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
    
    # 4. ROLLING VOLATILITY (realized volatility)
    # WHY: Volatility clustering - high vol periods predict high vol ahead (and option premiums)
    df['volatility_20'] = df['log_return_1'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # 5. PRICE MOMENTUM
    # WHY: Momentum persists in the short term
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # 6. VOLUME INDICATORS
    # WHY: Volume confirms price moves. High volume = more conviction
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # 7. BOLLINGER BAND POSITION
    # WHY: Measures where price is relative to recent range (mean reversion signal)
    bb_period = 20
    df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
    bb_std = df['Close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 8. TIME-BASED FEATURES
    # WHY: Market behavior differs by time of day and day of week
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    
    # 9. CANDLE PATTERNS
    # WHY: Candle body and shadow ratios signal buyer/seller strength
    df['candle_body'] = (df['Close'] - df['Open']) / df['Open']
    df['candle_range'] = (df['High'] - df['Low']) / df['Open']
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
    
    # 10. AVERAGE TRUE RANGE (ATR)
    # WHY: Measures volatility, useful for position sizing
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_ratio'] = true_range / df['atr_14']
    
    print(f"Created {len(df.columns)} features")
    
    return df


def create_labels(df, profit_threshold=0.003, horizon=3, transaction_cost=0.0005):
    """
    Create binary labels for profitable trades
    
    WHY: We define profitability as price moving up by threshold within horizon,
    minus transaction costs (brokerage, slippage, impact)
    
    PARAMETERS:
    - profit_threshold: 0.3% move (0.003) is realistic for ATM options
    - horizon: 3 candles ahead (15 minutes on 5-min chart)
    - transaction_cost: 0.05% for round-trip (buy + sell) - conservative estimate
    
    LOGIC:
    Label = 1 if max(future_return over next 3 candles) >= threshold + costs
    Label = 0 otherwise
    
    CRITICAL: We look at MAXIMUM return in horizon, not just return at horizon end.
    This is because in real trading we can exit anytime within the window.
    """
    
    print("Creating target labels...")
    
    df = df.copy()
    
    # Calculate forward returns for next N candles
    forward_returns = []
    for i in range(1, horizon + 1):
        forward_returns.append(df['Close'].shift(-i) / df['Close'] - 1)
    
    # Maximum return achievable in the next 'horizon' candles
    df['max_forward_return'] = pd.concat(forward_returns, axis=1).max(axis=1)
    
    # Subtract transaction costs from returns
    # WHY: In real trading, costs eat into profits. Model must predict profitability AFTER costs
    net_return = df['max_forward_return'] - transaction_cost
    
    # Binary label: 1 if profitable after costs, 0 otherwise
    df['target'] = (net_return >= profit_threshold).astype(int)
    
    # Drop rows where we can't calculate forward returns (last 'horizon' rows)
    df = df.iloc[:-horizon]
    
    print(f"Label distribution:")
    print(df['target'].value_counts())
    print(f"Positive class ratio: {df['target'].mean():.2%}")
    
    # IMPORTANT: If positive class is < 5%, the threshold might be too high
    if df['target'].mean() < 0.05:
        print("WARNING: Very few positive examples. Consider lowering profit_threshold.")
    
    return df


def prepare_train_test_split(df, test_size=0.2):
    """
    Split data into train/test using TIME-BASED split (not random)
    
    WHY: Random splits cause data leakage in time series. We must train on past,
    test on future, just like real trading.
    
    We use a simple train/test split rather than walk-forward for simplicity.
    In production, use walk-forward cross-validation.
    """
    
    print("Splitting data...")
    
    # Define feature columns (exclude target, intermediate calculations, and OHLCV)
    feature_cols = [
        'log_return_1', 'log_return_3', 'log_return_5',
        'rsi_14', 'distance_from_vwap', 'volatility_20',
        'momentum_5', 'momentum_10', 'volume_ratio',
        'bb_position', 'minute_of_day', 'day_of_week', 'hour',
        'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
        'atr_ratio'
    ]
    
    # Drop rows with NaN (from rolling calculations)
    df_clean = df.dropna(subset=feature_cols + ['target'])
    
    print(f"Cleaned data: {len(df_clean)} rows (dropped {len(df) - len(df_clean)} rows with NaN)")
    
    # Time-based split
    split_idx = int(len(df_clean) * (1 - test_size))
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Train set: {len(X_train)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test set: {len(X_test)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    print(f"Train positive class: {y_train.mean():.2%}")
    print(f"Test positive class: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare performance
    
    WHY: Different algorithms have different strengths. Logistic Regression is
    interpretable and fast. Random Forest captures non-linear patterns.
    
    We emphasize PRECISION over recall because:
    - False positives = losing trades (cost money)
    - False negatives = missed opportunities (no direct cost)
    - In trading, we prefer fewer, higher-quality signals
    """
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Standardize features
    # WHY: Logistic Regression requires standardization. Tree models don't, but it doesn't hurt.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # MODEL 1: Logistic Regression
    # WHY: Fast, interpretable, gives probability estimates, works well with standardized features
    print("\n1. Training Logistic Regression...")
    
    # Use class_weight='balanced' to handle class imbalance
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Gives more weight to minority class
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    models['logistic_regression'] = lr_model
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results['logistic_regression'] = evaluate_model(
        y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression"
    )
    
    # Feature importance (coefficients)
    print("\nTop 10 Most Important Features (Logistic Regression):")
    feature_importance_lr = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(feature_importance_lr.head(10))
    
    # MODEL 2: Random Forest
    # WHY: Captures non-linear relationships, robust to outliers, provides feature importance
    print("\n2. Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Prevent overfitting
        min_samples_split=20,  # Require minimum samples to split
        min_samples_leaf=10,  # Require minimum samples in leaf
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train_scaled, y_train)
    models['random_forest'] = rf_model
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results['random_forest'] = evaluate_model(
        y_test, y_pred_rf, y_pred_proba_rf, "Random Forest"
    )
    
    # Feature importance
    print("\nTop 10 Most Important Features (Random Forest):")
    feature_importance_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance_rf.head(10))
    
    # COMPARE MODELS
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Metric':<20} {'Logistic Regression':<20} {'Random Forest':<20}")
    print("-"*60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize():<20} {results['logistic_regression'][metric]:<20.4f} {results['random_forest'][metric]:<20.4f}")
    
    # Select best model based on PRECISION (most important for trading)
    if results['logistic_regression']['precision'] >= results['random_forest']['precision']:
        best_model_name = 'logistic_regression'
        best_model = models['logistic_regression']
        best_scaler = scaler
    else:
        best_model_name = 'random_forest'
        best_model = models['random_forest']
        best_scaler = scaler
    
    print(f"\nBest model: {best_model_name} (based on precision)")
    
    return best_model, best_scaler, models, results, best_model_name


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Comprehensive model evaluation
    
    WHY: We need multiple metrics because accuracy alone is misleading with imbalanced classes.
    - Precision: Of predicted profitable trades, how many actually were?
    - Recall: Of all profitable trades, how many did we catch?
    - F1: Harmonic mean of precision and recall
    
    For trading, PRECISION matters most (avoid false signals).
    """
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # F1 score (harmonic mean of precision and recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} <- MOST IMPORTANT FOR TRADING")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    print(f"                Predicted Neg    Predicted Pos")
    print(f"Actual Neg      {tn:<16} {fp:<16}")
    print(f"Actual Pos      {fn:<16} {tp:<16}")

    
    print("\nInterpretation:")
    print(f"- True Negatives (correct rejections): {cm[0][0]}")
    print(f"- False Positives (bad trades we took): {cm[0][1]} <- COSTLY")
    print(f"- False Negatives (good trades we missed): {cm[1][0]}")
    print(f"- True Positives (good trades we took): {cm[1][1]} <- PROFITABLE")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=['Not Profitable', 'Profitable'],
        zero_division=0
    )
)

    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def save_model(model, scaler, feature_cols, model_name='best_model'):
    """
    Save trained model and preprocessing objects to disk
    
    WHY: We need to persist the model for:
    - Future predictions on new data
    - Deployment to production trading system
    - Reproducibility and version control
    """
    
    print(f"\nSaving model and artifacts...")
    
    # Save model
    model_path = f'{model_name}_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = f'{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature columns (critical for consistent preprocessing)
    feature_path = f'{model_name}_features.pkl'
    joblib.dump(feature_cols, feature_path)
    print(f"Feature list saved to: {feature_path}")
    
    print("\nTo load and use the model later:")
    print(f"  model = joblib.load('{model_path}')")
    print(f"  scaler = joblib.load('{scaler_path}')")
    print(f"  features = joblib.load('{feature_path}')")


def main():
    """
    Main execution pipeline
    
    This orchestrates the entire ML workflow from data to deployment-ready model.
    """
    
    print("="*80)
    print("NSE OPTIONS PROFITABILITY PREDICTION - ML PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # STEP 1: FETCH DATA
    print("\nSTEP 1: FETCHING DATA")
    print("-"*80)
    
    # Try to load cached data first
    try:
        df = pd.read_csv('nifty_data.csv', index_col='Date', parse_dates=True)
        print("Loaded cached data from nifty_data.csv")
    except FileNotFoundError:
        # Fetch fresh data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = fetch_data(
        symbol='^NSEI',  # NIFTY 50
        start_date=start_date,
        end_date=end_date,
        save_path='nifty_data.csv'
    )

    if df is None:
        raise RuntimeError("Data fetch failed. Stopping pipeline.")

    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # STEP 2: CREATE FEATURES
    print("\nSTEP 2: FEATURE ENGINEERING")
    print("-"*80)
    df = create_features(df)
    
    # STEP 3: CREATE LABELS
    print("\nSTEP 3: LABEL CREATION")
    print("-"*80)
    df = create_labels(
        df,
        profit_threshold=0.003,  # 0.3% profit target
        horizon=3,  # 3 candles (15 minutes on 5-min chart)
        transaction_cost=0.0005  # 0.05% round-trip cost
    )
    
    # STEP 4: TRAIN/TEST SPLIT
    print("\nSTEP 4: TRAIN/TEST SPLIT")
    print("-"*80)
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(
        df,
        test_size=0.2
    )
    
    # STEP 5: TRAIN MODELS
    print("\nSTEP 5: MODEL TRAINING")
    print("-"*80)
    best_model, best_scaler, all_models, results, best_model_name = train_models(
        X_train, y_train, X_test, y_test
    )
    
    # STEP 6: SAVE MODEL
    print("\nSTEP 6: SAVING MODEL")
    print("-"*80)
    save_model(best_model, best_scaler, feature_cols, model_name=best_model_name)
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now()}")
    print(f"\nBest model: {best_model_name}")
    print(f"Test Precision: {results[best_model_name]['precision']:.4f}")
    print(f"Test Recall: {results[best_model_name]['recall']:.4f}")
    
    print("\nIMPORTANT NOTES FOR PRODUCTION:")
    print("1. This model uses synthetic/daily-converted data. Replace with real 5-min NSE data.")
    print("2. Backtest thoroughly with actual option prices (not spot price approximations).")
    print("3. Account for option Greeks, IV, bid-ask spreads in real trading.")
    print("4. Implement proper position sizing and risk management.")
    print("5. Monitor model performance and retrain regularly (concept drift).")
    print("6. Consider walk-forward optimization instead of single train/test split.")
    print("7. Add live data pipeline for real-time predictions.")
    
    return best_model, best_scaler, feature_cols, results


# ENTRY POINT
if __name__ == "__main__":
    """
    Run the complete pipeline
    
    To execute: python this_script.py
    
    Required packages:
        pip install pandas numpy scikit-learn scipy joblib nsepy
    """
    
    try:
        model, scaler, features, results = main()
        print("\n✓ SUCCESS: Model training complete!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*80)
