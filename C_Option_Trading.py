"""
NSE Options Signal Engine - Production Grade (FIXED VERSION)
A signal-only options trading system using Upstox API and ML model filtering

CRITICAL: This system generates signals ONLY. NO order placement. NO auto-trading.

Strategy: User selects BUYING or SELLING at startup (locked for session)
Market: NSE NIFTY 50 Options (CE/PE)
ML: Pre-trained model used as probability filter only

FIXES IN THIS VERSION:
- Robust LTP fetching with multiple fallback strategies
- Better API response debugging
- Improved error handling for Upstox API structure variations
"""

import os
import sys
import json
import time
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration - modify these values as needed"""
    
    # Upstox API Configuration
    UPSTOX_API_BASE = 'https://api.upstox.com/v2'
    
    # Instrument Configuration
    INDEX_INSTRUMENT_KEY = 'NSE_INDEX|Nifty 50'
    OPTION_SYMBOL = 'NIFTY'
    INSTRUMENT_MASTER_PATH = 'NSE.json'  # Download from Upstox
    
    # ML Model Paths
    MODEL_PATH = 'logistic_regression_model.pkl'
    SCALER_PATH = 'logistic_regression_scaler.pkl'
    FEATURES_PATH = 'logistic_regression_features.pkl'
    
    # Trading Parameters
    ML_THRESHOLD_BUYING = 0.65  # High confidence for buying
    ML_THRESHOLD_SELLING_MIN = 0.45  # Neutral zone for selling
    ML_THRESHOLD_SELLING_MAX = 0.55
    
    # Risk Management
    MAX_SIGNALS_PER_DAY = 10
    SIGNAL_COOLDOWN_SECONDS = 300  # 5 minutes between signals
    MARKET_CUTOFF_TIME = '15:00:00'  # No signals after this time
    
    # Polling
    POLLING_INTERVAL_SECONDS = 30  # Check every 30 seconds
    
    # Debug Mode
    DEBUG_API_RESPONSES = True  # Set to False in production
    
    # Feature Engineering Proxy Values
    FEATURE_DEFAULTS = {
        'log_return_1': 0.0,
        'log_return_3': 0.0,
        'log_return_5': 0.0,
        'rsi_14': 50.0,
        'distance_from_vwap': 0.0,
        'volatility_20': 0.15,
        'momentum_5': 0.0,
        'momentum_10': 0.0,
        'volume_ratio': 1.0,
        'bb_position': 0.5,
        'minute_of_day': 0,
        'day_of_week': 0,
        'hour': 0,
        'candle_body': 0.0,
        'candle_range': 0.0,
        'upper_shadow': 0.0,
        'lower_shadow': 0.0,
        'atr_ratio': 1.0
    }


# ============================================================================
# STRATEGY SELECTION
# ============================================================================

def select_strategy() -> str:
    """
    Prompt user to select trading strategy at startup
    Strategy is locked for entire session - no mid-session changes allowed
    
    Returns:
        str: 'BUYING' or 'SELLING'
    
    WHY: Different strategies have different signal criteria
    BUYING = Directional, high conviction
    SELLING = Theta decay, neutral probability
    """
    print("=" * 80)
    print("NSE OPTIONS SIGNAL ENGINE - STRATEGY SELECTION")
    print("=" * 80)
    print("\nAvailable Strategies:")
    print("1. BUYING  - Directional trades (calls/puts), high ML confidence required")
    print("2. SELLING - Premium selling, neutral ML probability preferred")
    print("\nWARNING: Strategy will be LOCKED for this session. Choose carefully.")
    
    while True:
        choice = input("\nSelect strategy (BUYING / SELLING): ").strip().upper()
        
        if choice in ['BUYING', 'SELLING']:
            print(f"\n✓ Strategy LOCKED: {choice}")
            print(f"  This cannot be changed until you restart the program.")
            print("=" * 80)
            return choice
        else:
            print("✗ Invalid input. Enter 'BUYING' or 'SELLING' exactly.")


# ============================================================================
# ML MODEL LOADING
# ============================================================================

def load_ml_model() -> Tuple[Optional[object], Optional[object], Optional[List[str]]]:
    """
    Load pre-trained ML model, scaler, and feature names
    
    Returns:
        Tuple of (model, scaler, feature_names) or (None, None, None) if loading fails
    
    WHY: ML model provides probability filter, not direct trading decisions
    Model failure is not fatal - system can run without ML (all signals blocked)
    """
    print("\n" + "-" * 80)
    print("LOADING ML MODEL")
    print("-" * 80)
    
    try:
        # Load model
        if not os.path.exists(Config.MODEL_PATH):
            print(f"✗ Model file not found: {Config.MODEL_PATH}")
            return None, None, None
        
        model = joblib.load(Config.MODEL_PATH)
        print(f"✓ Model loaded: {Config.MODEL_PATH}")
        
        # Load scaler
        if not os.path.exists(Config.SCALER_PATH):
            print(f"✗ Scaler file not found: {Config.SCALER_PATH}")
            return None, None, None
        
        scaler = joblib.load(Config.SCALER_PATH)
        print(f"✓ Scaler loaded: {Config.SCALER_PATH}")
        
        # Load feature names
        if not os.path.exists(Config.FEATURES_PATH):
            print(f"✗ Features file not found: {Config.FEATURES_PATH}")
            return None, None, None
        
        features = joblib.load(Config.FEATURES_PATH)
        print(f"✓ Features loaded: {len(features)} features")
        print(f"  Features: {features}")
        
        return model, scaler, features
        
    except Exception as e:
        print(f"✗ Error loading ML model: {e}")
        return None, None, None


# ============================================================================
# INSTRUMENT MASTER & EXPIRY RESOLUTION
# ============================================================================

def load_instrument_master() -> Optional[pd.DataFrame]:
    """
    Load Upstox Instrument Master (NSE.json) from local file
    
    Returns:
        DataFrame with instrument data or None if loading fails
    
    WHY: We need this to resolve valid NIFTY option expiries
    DO NOT hardcode expiry dates - they change weekly/monthly
    
    CRITICAL: Download NSE.json from Upstox API documentation first
    https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
    """
    print("\n" + "-" * 80)
    print("LOADING INSTRUMENT MASTER")
    print("-" * 80)
    
    if not os.path.exists(Config.INSTRUMENT_MASTER_PATH):
        print(f"✗ Instrument master not found: {Config.INSTRUMENT_MASTER_PATH}")
        print("  Download from: https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz")
        print("  Extract and place as NSE.json in the same directory")
        return None
    
    try:
        # Load JSON file
        with open(Config.INSTRUMENT_MASTER_PATH, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✓ Loaded {len(df)} instruments")
        
        # Filter for NIFTY options only
        nifty_options = df[
            (df['name'] == Config.OPTION_SYMBOL) & 
            (df['instrument_type'].isin(['CE', 'PE']))
        ].copy()
        
        print(f"✓ Found {len(nifty_options)} NIFTY option contracts")
        
        return nifty_options
        
    except Exception as e:
        print(f"✗ Error loading instrument master: {e}")
        return None


def resolve_expiry(instruments_df: pd.DataFrame) -> Optional[str]:
    """
    Resolve the nearest future expiry date from instrument master
    
    Args:
        instruments_df: DataFrame of NIFTY option instruments
    
    Returns:
        Expiry date string in 'YYYY-MM-DD' format or None
    
    WHY: Expiry dates are NOT hardcoded. They vary (weekly/monthly).
    System must dynamically find the next valid expiry.
    
    CRITICAL: Handle both epoch timestamps and date strings
    """
    print("\n" + "-" * 80)
    print("RESOLVING EXPIRY DATE")
    print("-" * 80)
    
    if instruments_df is None or len(instruments_df) == 0:
        print("✗ No instruments data available")
        return None
    
    try:
        # Extract expiry column
        if 'expiry' not in instruments_df.columns:
            print("✗ 'expiry' column not found in instrument master")
            return None
        
        expiries = instruments_df['expiry'].unique()
        print(f"  Found {len(expiries)} unique expiry values")
        
        # Normalize expiry values (handle epoch timestamps and date strings)
        expiry_dates = []
        
        for exp in expiries:
            try:
                # Skip null/empty
                if pd.isna(exp) or exp == '':
                    continue
                
                # Handle epoch timestamp (milliseconds)
                if isinstance(exp, (int, float)):
                    dt = datetime.fromtimestamp(exp / 1000)
                    expiry_dates.append(dt.date())
                
                # Handle string date
                elif isinstance(exp, str):
                    # Try multiple date formats
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            dt = datetime.strptime(exp, fmt)
                            expiry_dates.append(dt.date())
                            break
                        except ValueError:
                            continue
            
            except Exception as e:
                # Skip problematic expiry values
                continue
        
        if not expiry_dates:
            print("✗ No valid expiry dates found")
            return None
        
        # Filter future expiries only
        today = datetime.now().date()
        future_expiries = sorted([exp for exp in expiry_dates if exp > today])
        
        if not future_expiries:
            print("✗ No future expiries available")
            return None
        
        # Select nearest expiry
        nearest_expiry = future_expiries[0]
        expiry_str = nearest_expiry.strftime('%Y-%m-%d')
        
        print(f"✓ Nearest expiry: {expiry_str}")
        print(f"  Days until expiry: {(nearest_expiry - today).days}")
        
        return expiry_str
        
    except Exception as e:
        print(f"✗ Error resolving expiry: {e}")
        return None


# ============================================================================
# UPSTOX API INTERACTIONS (FIXED WITH DEBUGGING)
# ============================================================================

def debug_print_api_response(response_data: Dict, prefix: str = ""):
    """
    Helper function to print API response structure for debugging
    
    WHY: Upstox API response structures can vary, and debugging helps identify issues
    """
    if not Config.DEBUG_API_RESPONSES:
        return
    
    print(f"\n{prefix}API Response Structure:")
    print(f"{prefix}Type: {type(response_data)}")
    
    if isinstance(response_data, dict):
        print(f"{prefix}Keys: {list(response_data.keys())}")
        for key in response_data.keys():
            print(f"{prefix}  {key}: {type(response_data[key])}")
            if isinstance(response_data[key], dict) and len(response_data[key]) < 5:
                print(f"{prefix}    Sub-keys: {list(response_data[key].keys())}")


def fetch_nifty_ltp() -> Optional[float]:
    """
    Fetch NIFTY 50 index Last Traded Price using Upstox API
    
    ROBUST VERSION with multiple fallback strategies:
    1. Try /market-quote/quotes endpoint
    2. Try /market-quote/ltp endpoint
    3. Parse response with multiple path strategies
    
    Returns:
        float: NIFTY LTP or None if all strategies fail
    
    WHY: Upstox API response structure may vary, we need robust parsing
    """
    if not UPSTOX_ACCESS_TOKEN:
        print("✗ UPSTOX_ACCESS_TOKEN not set. Cannot fetch data.")
        return None
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
    }
    
    # ========================================================================
    # STRATEGY 1: Try /market-quote/quotes endpoint
    # ========================================================================
    try:
        print("  Trying /market-quote/quotes endpoint...")
        url = f"{Config.UPSTOX_API_BASE}/market-quote/quotes"
        params = {'instrument_key': Config.INDEX_INSTRUMENT_KEY}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            debug_print_api_response(data, "  ")
            
            # Try multiple parsing paths
            ltp = None
            
            # Path 1: data -> {instrument_key} -> last_price
            if 'data' in data:
                if isinstance(data['data'], dict):
                    if Config.INDEX_INSTRUMENT_KEY in data['data']:
                        instrument_data = data['data'][Config.INDEX_INSTRUMENT_KEY]
                        if isinstance(instrument_data, dict):
                            ltp = instrument_data.get('last_price') or instrument_data.get('ltp')
            
            # Path 2: data -> last_price (direct)
            if ltp is None and 'data' in data:
                if isinstance(data['data'], dict):
                    ltp = data['data'].get('last_price') or data['data'].get('ltp')
            
            # Path 3: Direct last_price/ltp at root
            if ltp is None:
                ltp = data.get('last_price') or data.get('ltp')
            
            if ltp is not None:
                print(f"  ✓ Found LTP via /quotes: {ltp}")
                return float(ltp)
        else:
            print(f"  ✗ /quotes returned {response.status_code}")
    
    except Exception as e:
        print(f"  ✗ /quotes strategy failed: {e}")
    
    # ========================================================================
    # STRATEGY 2: Try /market-quote/ltp endpoint
    # ========================================================================
    try:
        print("  Trying /market-quote/ltp endpoint...")
        url = f"{Config.UPSTOX_API_BASE}/market-quote/ltp"
        params = {'instrument_key': Config.INDEX_INSTRUMENT_KEY}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            debug_print_api_response(data, "  ")
            
            # Try multiple parsing paths
            ltp = None
            
            # Path 1: data -> {instrument_key} -> last_price
            if 'data' in data:
                if isinstance(data['data'], dict):
                    if Config.INDEX_INSTRUMENT_KEY in data['data']:
                        instrument_data = data['data'][Config.INDEX_INSTRUMENT_KEY]
                        if isinstance(instrument_data, dict):
                            ltp = instrument_data.get('last_price') or instrument_data.get('ltp')
            
            # Path 2: data -> last_price (direct)
            if ltp is None and 'data' in data:
                if isinstance(data['data'], dict):
                    ltp = data['data'].get('last_price') or data['data'].get('ltp')
            
            # Path 3: Direct last_price/ltp at root
            if ltp is None:
                ltp = data.get('last_price') or data.get('ltp')
            
            if ltp is not None:
                print(f"  ✓ Found LTP via /ltp: {ltp}")
                return float(ltp)
        else:
            print(f"  ✗ /ltp returned {response.status_code}")
            if Config.DEBUG_API_RESPONSES:
                print(f"    Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"  ✗ /ltp strategy failed: {e}")
    
    # ========================================================================
    # ALL STRATEGIES FAILED
    # ========================================================================
    print("✗ All LTP fetch strategies failed")
    print("  Please check:")
    print("    1. UPSTOX_ACCESS_TOKEN is valid and not expired")
    print("    2. API is accessible (not rate limited)")
    print("    3. Instrument key is correct: " + Config.INDEX_INSTRUMENT_KEY)
    
    return None


def fetch_option_chain(expiry_date: str) -> Optional[Dict]:
    """
    Fetch NIFTY option chain data from Upstox API
    
    Args:
        expiry_date: Expiry date in 'YYYY-MM-DD' format
    
    Returns:
        Dict with option chain data or None if fetch fails
    
    WHY: Option chain provides strike prices, Greeks, IV, premiums
    This is our PRIMARY data source for option selection
    """
    if not UPSTOX_ACCESS_TOKEN:
        print("✗ UPSTOX_ACCESS_TOKEN not set. Cannot fetch option chain.")
        return None
    
    if not expiry_date:
        print("✗ No expiry date provided. Cannot fetch option chain.")
        return None
    
    try:
        url = f"{Config.UPSTOX_API_BASE}/option/chain"
        params = {
            'instrument_key': Config.INDEX_INSTRUMENT_KEY,
            'expiry_date': expiry_date
        }
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"✗ Option chain API error: {response.status_code}")
            if Config.DEBUG_API_RESPONSES:
                print(f"  Response: {response.text[:300]}")
            return None
        
        data = response.json()
        debug_print_api_response(data, "  ")
        
        if 'data' not in data:
            print("✗ Invalid option chain response structure")
            return None
        
        return data['data']
        
    except Exception as e:
        print(f"✗ Error fetching option chain: {e}")
        return None


# ============================================================================
# OPTION SELECTION LOGIC
# ============================================================================

def select_atm_options(option_chain: Dict, nifty_ltp: float) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Select ATM CALL and ATM PUT from option chain
    
    Args:
        option_chain: Option chain data from Upstox API
        nifty_ltp: Current NIFTY spot price
    
    Returns:
        Tuple of (ATM_CALL, ATM_PUT) option dictionaries or (None, None)
    
    WHY: ATM options have highest liquidity and respond most to price movement
    
    CRITICAL: Handle missing data gracefully - skip if legs are missing
    """
    if not option_chain or not nifty_ltp:
        return None, None
    
    try:
        # Option chain structure: list of strike data
        chain_data = option_chain if isinstance(option_chain, list) else option_chain.get('data', [])
        
        if not chain_data:
            print("✗ Empty option chain data")
            return None, None
        
        # Find ATM strike (closest to NIFTY LTP)
        strikes = []
        for item in chain_data:
            strike = item.get('strike_price') or item.get('strike')
            if strike:
                strikes.append(strike)
        
        if not strikes:
            print("✗ No strikes found in option chain")
            return None, None
        
        # Find closest strike to spot
        atm_strike = min(strikes, key=lambda x: abs(x - nifty_ltp))
        print(f"  ATM Strike: {atm_strike} (NIFTY: {nifty_ltp:.2f})")
        
        # Extract ATM CALL and ATM PUT
        atm_call = None
        atm_put = None
        
        for item in chain_data:
            item_strike = item.get('strike_price') or item.get('strike')
            if item_strike == atm_strike:
                # Check for CALL data
                if 'call_options' in item and item['call_options']:
                    atm_call = item['call_options']
                    atm_call['strike'] = atm_strike
                
                # Check for PUT data
                if 'put_options' in item and item['put_options']:
                    atm_put = item['put_options']
                    atm_put['strike'] = atm_strike
        
        # Validate
        if atm_call:
            print(f"  ✓ ATM CALL found: Strike {atm_strike}")
        else:
            print(f"  ✗ ATM CALL missing")
        
        if atm_put:
            print(f"  ✓ ATM PUT found: Strike {atm_strike}")
        else:
            print(f"  ✗ ATM PUT missing")
        
        return atm_call, atm_put
        
    except Exception as e:
        print(f"✗ Error selecting ATM options: {e}")
        return None, None


# ============================================================================
# FEATURE ENGINEERING (PROXY FROM OPTION CHAIN)
# ============================================================================

def build_features_from_option_chain(
    option_data: Dict,
    nifty_ltp: float,
    ml_features: List[str]
) -> Optional[np.ndarray]:
    """
    Build feature vector for ML model using option chain data as proxy
    
    Args:
        option_data: ATM option data (CALL or PUT)
        nifty_ltp: Current NIFTY spot price
        ml_features: List of feature names expected by ML model
    
    Returns:
        numpy array of features or None if construction fails
    
    WHY: We don't have historical candles. We use option chain data as proxies:
    - IV → volatility proxy
    - Greeks → momentum/direction proxy
    - Bid/Ask spread → liquidity proxy
    
    CRITICAL: Feature alignment must match training data exactly
    """
    if not option_data or not ml_features:
        return None
    
    try:
        # Extract option chain fields (defensive extraction)
        ltp = option_data.get('last_price', 0) or option_data.get('ltp', 0) or 0
        bid = option_data.get('bid_price', 0) or option_data.get('bid', 0) or 0
        ask = option_data.get('ask_price', 0) or option_data.get('ask', 0) or 0
        iv = option_data.get('implied_volatility', 0.15) or option_data.get('iv', 0.15) or 0.15
        delta = option_data.get('delta', 0) or 0
        theta = option_data.get('theta', 0) or 0
        gamma = option_data.get('gamma', 0) or 0
        vega = option_data.get('vega', 0) or 0
        
        # Time-based features
        now = datetime.now()
        minute_of_day = now.hour * 60 + now.minute
        day_of_week = now.weekday()
        hour = now.hour
        
        # Build feature dictionary
        feature_dict = Config.FEATURE_DEFAULTS.copy()
        
        # Update with option-derived proxies
        feature_dict['volatility_20'] = iv  # Use IV as volatility proxy
        feature_dict['minute_of_day'] = minute_of_day
        feature_dict['day_of_week'] = day_of_week
        feature_dict['hour'] = hour
        
        # Momentum proxy (delta is directional sensitivity)
        feature_dict['momentum_5'] = delta / 100 if delta else 0
        feature_dict['momentum_10'] = delta / 100 if delta else 0
        
        # Spread-based features
        if ask > bid and bid > 0:
            spread_pct = (ask - bid) / bid
            feature_dict['candle_range'] = spread_pct
        
        # RSI proxy (use normalized delta)
        # Delta ranges -1 to +1, RSI ranges 0 to 100
        if delta:
            rsi_proxy = 50 + (delta * 50)  # Map delta to RSI-like scale
            feature_dict['rsi_14'] = max(0, min(100, rsi_proxy))
        
        # Build feature array in exact order expected by model
        feature_array = []
        for feat_name in ml_features:
            if feat_name in feature_dict:
                feature_array.append(feature_dict[feat_name])
            else:
                # Missing feature - use zero (defensive)
                print(f"  Warning: Feature '{feat_name}' not in proxy dict, using 0")
                feature_array.append(0.0)
        
        return np.array(feature_array).reshape(1, -1)
        
    except Exception as e:
        print(f"✗ Error building features: {e}")
        return None


# ============================================================================
# SIGNAL GENERATION LOGIC
# ============================================================================

def generate_signal(
    strategy: str,
    atm_call: Dict,
    atm_put: Dict,
    nifty_ltp: float,
    model: object,
    scaler: object,
    ml_features: List[str]
) -> Dict:
    """
    Generate trading signal based on strategy and ML model
    
    Args:
        strategy: 'BUYING' or 'SELLING'
        atm_call: ATM CALL option data
        atm_put: ATM PUT option data
        nifty_ltp: Current NIFTY spot price
        model: Trained ML model
        scaler: Feature scaler
        ml_features: List of feature names
    
    Returns:
        Dict with signal details:
        {
            'action': 'BUY_CALL' / 'BUY_PUT' / 'SELL_CALL' / 'SELL_PUT' / 'NO_TRADE',
            'strike': float,
            'option_type': 'CE' / 'PE',
            'probability': float,
            'reason': str,
            'premium': float,
            'greeks': dict
        }
    
    WHY: Different strategies have different signal criteria
    BUYING: High ML confidence, strong delta, expanding IV
    SELLING: Neutral ML, high theta decay, elevated IV
    """
    signal = {
        'action': 'NO_TRADE',
        'strike': None,
        'option_type': None,
        'probability': 0.0,
        'reason': 'Not evaluated',
        'premium': 0.0,
        'greeks': {}
    }
    
    # Validate inputs
    if not model or not scaler or not ml_features:
        signal['reason'] = 'ML model not available'
        return signal
    
    if not atm_call and not atm_put:
        signal['reason'] = 'No ATM options available'
        return signal
    
    # Evaluate CALL option
    call_probability = 0.0
    call_features = None
    
    if atm_call:
        call_features = build_features_from_option_chain(atm_call, nifty_ltp, ml_features)
        
        if call_features is not None:
            try:
                call_features_scaled = scaler.transform(call_features)
                call_probability = model.predict_proba(call_features_scaled)[0][1]
            except Exception as e:
                print(f"  ✗ Error predicting CALL: {e}")
                call_probability = 0.0
    
    # Evaluate PUT option
    put_probability = 0.0
    put_features = None
    
    if atm_put:
        put_features = build_features_from_option_chain(atm_put, nifty_ltp, ml_features)
        
        if put_features is not None:
            try:
                put_features_scaled = scaler.transform(put_features)
                put_probability = model.predict_proba(put_features_scaled)[0][1]
            except Exception as e:
                print(f"  ✗ Error predicting PUT: {e}")
                put_probability = 0.0
    
    print(f"  ML Probabilities - CALL: {call_probability:.3f}, PUT: {put_probability:.3f}")
    
    # ========================================================================
    # STRATEGY: BUYING (Directional, High Confidence)
    # ========================================================================
    if strategy == 'BUYING':
        # Criteria:
        # 1. ML probability > threshold
        # 2. Strong delta (directional bias)
        # 3. IV expanding (momentum)
        
        # Check CALL
        if atm_call and call_probability >= Config.ML_THRESHOLD_BUYING:
            delta = atm_call.get('delta', 0) or 0
            iv = atm_call.get('implied_volatility', 0) or atm_call.get('iv', 0) or 0
            
            if delta > 0.4 and iv > 0.10:  # Strong bullish delta, IV > 10%
                signal['action'] = 'BUY_CALL'
                signal['strike'] = atm_call['strike']
                signal['option_type'] = 'CE'
                signal['probability'] = call_probability
                signal['premium'] = atm_call.get('last_price', 0) or atm_call.get('ltp', 0)
                signal['greeks'] = {
                    'delta': delta,
                    'theta': atm_call.get('theta', 0),
                    'iv': iv
                }
                signal['reason'] = f'High ML confidence ({call_probability:.2%}), strong delta ({delta:.3f})'
                return signal
        
        # Check PUT
        if atm_put and put_probability >= Config.ML_THRESHOLD_BUYING:
            delta = abs(atm_put.get('delta', 0) or 0)  # PUT delta is negative, use abs
            iv = atm_put.get('implied_volatility', 0) or atm_put.get('iv', 0) or 0
            
            if delta > 0.4 and iv > 0.10:  # Strong bearish delta
                signal['action'] = 'BUY_PUT'
                signal['strike'] = atm_put['strike']
                signal['option_type'] = 'PE'
                signal['probability'] = put_probability
                signal['premium'] = atm_put.get('last_price', 0) or atm_put.get('ltp', 0)
                signal['greeks'] = {
                    'delta': atm_put.get('delta', 0),
                    'theta': atm_put.get('theta', 0),
                    'iv': iv
                }
                signal['reason'] = f'High ML confidence ({put_probability:.2%}), strong delta ({delta:.3f})'
                return signal
        
        signal['reason'] = 'No BUYING opportunities (low ML confidence or weak delta)'
        return signal
    
    # ========================================================================
    # STRATEGY: SELLING (Theta Decay, Neutral Probability)
    # ========================================================================
    elif strategy == 'SELLING':
        # Criteria:
        # 1. ML probability near neutral (0.45 - 0.55)
        # 2. Strong negative theta (time decay)
        # 3. Elevated IV (high premium)
        # 4. Delta near neutral (range-bound expectation)
        
        # Check CALL
        if atm_call:
            if Config.ML_THRESHOLD_SELLING_MIN <= call_probability <= Config.ML_THRESHOLD_SELLING_MAX:
                delta = abs(atm_call.get('delta', 0) or 0)
                theta = atm_call.get('theta', 0) or 0
                iv = atm_call.get('implied_volatility', 0) or atm_call.get('iv', 0) or 0
                
                # Want: low delta (neutral), negative theta (decay), high IV (premium)
                if delta < 0.35 and theta < -5 and iv > 0.15:
                    signal['action'] = 'SELL_CALL'
                    signal['strike'] = atm_call['strike']
                    signal['option_type'] = 'CE'
                    signal['probability'] = call_probability
                    signal['premium'] = atm_call.get('last_price', 0) or atm_call.get('ltp', 0)
                    signal['greeks'] = {
                        'delta': atm_call.get('delta', 0),
                        'theta': theta,
                        'iv': iv
                    }
                    signal['reason'] = f'Neutral ML ({call_probability:.2%}), theta decay ({theta:.2f}), high IV ({iv:.2%})'
                    return signal
        
        # Check PUT
        if atm_put:
            if Config.ML_THRESHOLD_SELLING_MIN <= put_probability <= Config.ML_THRESHOLD_SELLING_MAX:
                delta = abs(atm_put.get('delta', 0) or 0)
                theta = atm_put.get('theta', 0) or 0
                iv = atm_put.get('implied_volatility', 0) or atm_put.get('iv', 0) or 0
                
                if delta < 0.35 and theta < -5 and iv > 0.15:
                    signal['action'] = 'SELL_PUT'
                    signal['strike'] = atm_put['strike']
                    signal['option_type'] = 'PE'
                    signal['probability'] = put_probability
                    signal['premium'] = atm_put.get('last_price', 0) or atm_put.get('ltp', 0)
                    signal['greeks'] = {
                        'delta': atm_put.get('delta', 0),
                        'theta': theta,
                        'iv': iv
                    }
                    signal['reason'] = f'Neutral ML ({put_probability:.2%}), theta decay ({theta:.2f}), high IV ({iv:.2%})'
                    return signal
        
        signal['reason'] = 'No SELLING opportunities (ML not neutral or Greeks unfavorable)'
        return signal
    
    signal['reason'] = 'Unknown strategy'
    return signal


# ============================================================================
# RISK FILTERS
# ============================================================================

class RiskManager:
    """
    Enforces risk rules and trading limits
    
    WHY: Prevent overtrading, bad timing, and excessive signals
    """
    
    def __init__(self):
        self.signals_today = 0
        self.last_signal_time = None
    
    def reset_daily(self):
        """Reset daily counters (call at market open)"""
        self.signals_today = 0
        self.last_signal_time = None
    
    def can_generate_signal(self) -> Tuple[bool, str]:
        """
        Check if system is allowed to generate a signal now
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check daily limit
        if self.signals_today >= Config.MAX_SIGNALS_PER_DAY:
            return False, f'Daily signal limit reached ({Config.MAX_SIGNALS_PER_DAY})'
        
        # Check cooldown
        if self.last_signal_time:
            elapsed = (datetime.now() - self.last_signal_time).total_seconds()
            if elapsed < Config.SIGNAL_COOLDOWN_SECONDS:
                remaining = Config.SIGNAL_COOLDOWN_SECONDS - elapsed
                return False, f'Cooldown active ({remaining:.0f}s remaining)'
        
        # Check time cutoff
        now = datetime.now().time()
        cutoff = datetime.strptime(Config.MARKET_CUTOFF_TIME, '%H:%M:%S').time()
        
        if now >= cutoff:
            return False, f'After market cutoff ({Config.MARKET_CUTOFF_TIME})'
        
        return True, 'OK'
    
    def record_signal(self):
        """Record that a signal was generated"""
        self.signals_today += 1
        self.last_signal_time = datetime.now()


# ============================================================================
# MAIN LOOP
# ============================================================================

def main_loop():
    """
    Main signal generation loop
    
    Workflow:
    1. Select and lock strategy
    2. Load ML model
    3. Resolve expiry
    4. Poll option chain
    5. Generate signals
    6. Apply risk filters
    7. Display signals
    8. Repeat
    """
    print("=" * 80)
    print("NSE OPTIONS SIGNAL ENGINE - STARTING (FIXED VERSION)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: STRATEGY SELECTION (LOCKED FOR SESSION)
    # ========================================================================
    strategy = select_strategy()
    
    # ========================================================================
    # STEP 2: LOAD ML MODEL
    # ========================================================================
    model, scaler, ml_features = load_ml_model()
    
    if not model:
        print("\n✗ CRITICAL: ML model failed to load. System cannot generate signals.")
        print("  Please ensure model files exist:")
        print(f"    - {Config.MODEL_PATH}")
        print(f"    - {Config.SCALER_PATH}")
        print(f"    - {Config.FEATURES_PATH}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: LOAD INSTRUMENT MASTER & RESOLVE EXPIRY
    # ========================================================================
    instruments_df = load_instrument_master()
    
    if instruments_df is None:
        print("\n✗ CRITICAL: Instrument master failed to load.")
        print("  System cannot resolve expiry dates without this file.")
        sys.exit(1)
    
    expiry_date = resolve_expiry(instruments_df)
    
    if not expiry_date:
        print("\n✗ CRITICAL: Could not resolve valid expiry date.")
        print("  System cannot fetch option chain without expiry.")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: INITIALIZE RISK MANAGER
    # ========================================================================
    risk_mgr = RiskManager()
    
    # ========================================================================
    # STEP 5: MAIN POLLING LOOP
    # ========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL ENGINE ACTIVE")
    print("=" * 80)
    print(f"Strategy: {strategy}")
    print(f"Expiry: {expiry_date}")
    print(f"Polling interval: {Config.POLLING_INTERVAL_SECONDS}s")
    print(f"Daily signal limit: {Config.MAX_SIGNALS_PER_DAY}")
    print(f"Signal cooldown: {Config.SIGNAL_COOLDOWN_SECONDS}s")
    print(f"Market cutoff: {Config.MARKET_CUTOFF_TIME}")
    print(f"Debug mode: {Config.DEBUG_API_RESPONSES}")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration} | {timestamp}")
            print(f"Strategy: {strategy} | Signals Today: {risk_mgr.signals_today}/{Config.MAX_SIGNALS_PER_DAY}")
            print(f"{'=' * 80}")
            
            # Check risk filters BEFORE doing any work
            can_signal, reason = risk_mgr.can_generate_signal()
            
            if not can_signal:
                print(f"⏸  Signal generation blocked: {reason}")
                print(f"   Waiting {Config.POLLING_INTERVAL_SECONDS}s...")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)
                continue
            
            # ================================================================
            # FETCH NIFTY LTP
            # ================================================================
            print("\n→ Fetching NIFTY LTP...")
            nifty_ltp = fetch_nifty_ltp()
            
            if not nifty_ltp:
                print("✗ Failed to fetch NIFTY LTP. Skipping this iteration.")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)
                continue
            
            print(f"✓ NIFTY LTP: {nifty_ltp:.2f}")
            
            # ================================================================
            # FETCH OPTION CHAIN
            # ================================================================
            print(f"\n→ Fetching option chain (Expiry: {expiry_date})...")
            option_chain = fetch_option_chain(expiry_date)
            
            if not option_chain:
                print("✗ Failed to fetch option chain. Skipping this iteration.")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)
                continue
            
            print("✓ Option chain fetched")
            
            # ================================================================
            # SELECT ATM OPTIONS
            # ================================================================
            print("\n→ Selecting ATM options...")
            atm_call, atm_put = select_atm_options(option_chain, nifty_ltp)
            
            if not atm_call and not atm_put:
                print("✗ No ATM options available. Skipping this iteration.")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)
                continue
            
            # ================================================================
            # GENERATE SIGNAL
            # ================================================================
            print("\n→ Generating signal...")
            signal = generate_signal(
                strategy=strategy,
                atm_call=atm_call,
                atm_put=atm_put,
                nifty_ltp=nifty_ltp,
                model=model,
                scaler=scaler,
                ml_features=ml_features
            )
            
            # ================================================================
            # DISPLAY SIGNAL
            # ================================================================
            print("\n" + "=" * 80)
            print("SIGNAL OUTPUT")
            print("=" * 80)
            print(f"Timestamp:    {timestamp}")
            print(f"Strategy:     {strategy}")
            print(f"Action:       {signal['action']}")
            print(f"Strike:       {signal['strike']}")
            print(f"Option Type:  {signal['option_type']}")
            print(f"Probability:  {signal['probability']:.2%}")
            print(f"Premium:      ₹{signal['premium']:.2f}")
            print(f"Reason:       {signal['reason']}")
            
            if signal['greeks']:
                print(f"Greeks:")
                for key, val in signal['greeks'].items():
                    print(f"  {key}: {val}")
            
            print("=" * 80)
            
            # Record signal if action was taken
            if signal['action'] != 'NO_TRADE':
                risk_mgr.record_signal()
                print(f"\n✓ Signal recorded. Total today: {risk_mgr.signals_today}")
            
            # ================================================================
            # WAIT BEFORE NEXT ITERATION
            # ================================================================
            print(f"\n⏳ Waiting {Config.POLLING_INTERVAL_SECONDS}s until next check...")
            time.sleep(Config.POLLING_INTERVAL_SECONDS)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("SIGNAL ENGINE STOPPED BY USER")
        print("=" * 80)
        print(f"Total signals generated today: {risk_mgr.signals_today}")
        print(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for NSE Options Signal Engine
    
    PREREQUISITES:
    1. Set UPSTOX_ACCESS_TOKEN environment variable
    2. Download NSE.json from Upstox (place in same directory)
    3. Train ML model (model, scaler, features .pkl files)
    
    USAGE:
    export UPSTOX_ACCESS_TOKEN="your_token_here"
    python C_Option_Trading_FIXED.py
    
    SAFETY:
    - This system generates SIGNALS ONLY
    - NO order placement
    - NO auto-trading
    - Human must execute trades manually
    """
    
    # Validate prerequisites
    if not UPSTOX_ACCESS_TOKEN:
        print("=" * 80)
        print("ERROR: UPSTOX_ACCESS_TOKEN not set")
        print("=" * 80)
        print("Please set your Upstox access token:")
        print("  export UPSTOX_ACCESS_TOKEN='your_token_here'")
        print("\nOr set it in .env file:")
        print("  UPSTOX_ACCESS_TOKEN=your_token_here")
        print("=" * 80)
        sys.exit(1)
    
    # Run main loop
    main_loop()