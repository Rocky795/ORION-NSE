import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math

# --- CONFIGURATION ---
class Config:
    # Nifty Specifics
    STRIKE_INTERVAL = 50 
    
    # Layer-1 Thresholds (The Gatekeeper)
    MODEL_CONFIDENCE_THRESHOLD = 0.60  # Minimum prob to engage
    HIGH_CONFIDENCE_THRESHOLD = 0.85   # Prob required for OTM buys

    # IV Thresholds (Percentile)
    IV_LOW_PERCENTILE = 30
    IV_HIGH_PERCENTILE = 70

    # Expiry Thresholds (Days)
    NEAR_EXPIRY_DAYS = 2
    FAR_EXPIRY_DAYS = 5

class OptionExecutionEngine:
    def __init__(self):
        self.config = Config()

    def _get_days_to_expiry(self, expiry_date_str: str) -> int:
        """Calculates days remaining to expiry."""
        # Format expectation: 'YYYY-MM-DD'
        expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        today = datetime.now()
        delta = expiry - today
        return max(0, delta.days + 1) # +1 to count today

    def _calculate_iv_percentile(self, current_iv: float, iv_history: list) -> float:
        """
        Computes IV Rank/Percentile based on historical window.
        Input: current_iv (float), iv_history (list of floats from last N days)
        """
        if not iv_history:
            return 50.0 # Default fallback
        
        iv_history = np.array(iv_history)
        return (np.sum(iv_history < current_iv) / len(iv_history)) * 100

    def _select_strike(self, spot_price: float, direction: str, action: str, model_prob: float) -> int:
        """
        Selects strike price based on:
        1. ATM logic (Rounding spot to nearest interval)
        2. Model probability (Aggressive OTM vs Safe ATM)
        3. Buy vs Sell logic
        """
        # 1. Identify ATM Strike
        atm_strike = round(spot_price / self.config.STRIKE_INTERVAL) * self.config.STRIKE_INTERVAL
        
        # 2. Determine Steps (0 = ATM, 1 = 1 Strike OTM, etc.)
        strike_step = 0

        if action == "BUY_OPTION":
            # Constraint 3: Prefer ATM, OTM only if prob > 0.85
            if model_prob > self.config.HIGH_CONFIDENCE_THRESHOLD:
                strike_step = 1 # Go 1 strike OTM for cheaper premium/higher gamma
            else:
                strike_step = 0 # Stay ATM
        
        elif action == "SELL_OPTION":
            # Constraint 3: Prefer ATM or 1-step OTM
            strike_step = 1 # Prefer selling OTM for safety buffer

        # 3. Calculate Final Strike based on Direction
        if direction == "BULLISH":
            # For CE: OTM is higher strike
            selected_strike = atm_strike + (strike_step * self.config.STRIKE_INTERVAL)
        elif direction == "BEARISH":
            # For PE: OTM is lower strike
            selected_strike = atm_strike - (strike_step * self.config.STRIKE_INTERVAL)
        else:
            selected_strike = atm_strike

        return int(selected_strike)

    def _decide_strategy_action(self, iv_percentile: float, days_to_expiry: int, direction: str) -> str:
        """
        Constraint 4: Buy vs Sell Premium Decision Table
        """
        is_high_iv = iv_percentile > self.config.IV_HIGH_PERCENTILE
        is_low_iv = iv_percentile < self.config.IV_LOW_PERCENTILE
        is_near_expiry = days_to_expiry <= self.config.NEAR_EXPIRY_DAYS

        # Logic Table
        if direction == "NEUTRAL":
            return "NO_TRADE"
            
        if is_high_iv and is_near_expiry:
            # High premium + Theta decay acceleration -> Sell
            return "SELL_OPTION"
        
        if is_low_iv:
            # Cheap premium -> Buy
            return "BUY_OPTION"
        
        # Default Logic for Mid-IV scenarios (Standard Directional Play)
        return "BUY_OPTION"

    def execute_logic(self, layer1_output: dict, market_data: dict) -> dict:
        """
        Main execution entry point.
        """
        # --- 1. GATEKEEPER (Layer 1 Check) ---
        direction = layer1_output.get("direction")
        prob = layer1_output.get("probability", 0.0)

        if prob < self.config.MODEL_CONFIDENCE_THRESHOLD:
            return {
                "trade_decision": "NO_TRADE",
                "option_type": None,
                "strike": None,
                "reason": f"Model probability ({prob:.2f}) below threshold ({self.config.MODEL_CONFIDENCE_THRESHOLD})"
            }

        if direction == "NEUTRAL":
            return {
                "trade_decision": "NO_TRADE",
                "option_type": None,
                "strike": None,
                "reason": "Model indicates Neutral direction. No directional trade."
            }

        # --- 2. MARKET CONTEXT ---
        spot = market_data['spot_price']
        current_iv = market_data['current_iv']
        iv_hist = market_data['iv_history']
        expiry_date = market_data['expiry_date']

        iv_percentile = self._calculate_iv_percentile(current_iv, iv_hist)
        dte = self._get_days_to_expiry(expiry_date)

        # --- 3. STRATEGY DECISION ---
        action = self._decide_strategy_action(iv_percentile, dte, direction)

        if action == "NO_TRADE":
            return {
                "trade_decision": "NO_TRADE",
                "option_type": None,
                "strike": None,
                "reason": "Market conditions (IV/Expiry) unfit for trade logic."
            }

        # --- 4. ASSET SELECTION ---
        # Map Direction to Option Type
        # Bullish Buy -> CE, Bullish Sell -> PE (Put Credit Spread/Short Put)
        # Bearish Buy -> PE, Bearish Sell -> CE (Call Credit Spread/Short Call)
        
        option_type = None
        if action == "BUY_OPTION":
            option_type = "CE" if direction == "BULLISH" else "PE"
        elif action == "SELL_OPTION":
            # If we sell to get bullish exposure, we sell Puts.
            # If we sell to get bearish exposure, we sell Calls.
            option_type = "PE" if direction == "BULLISH" else "CE"

        # Select Strike
        strike = self._select_strike(spot, direction, action, prob)

        # --- 5. FINAL OUTPUT CONSTRUCTION ---
        return {
            "trade_decision": action,
            "option_type": option_type,
            "strike": strike,
            "reason": (
                f"Dir: {direction} ({prob:.2f}), "
                f"IV%: {iv_percentile:.1f} (High>{self.config.IV_HIGH_PERCENTILE}, Low<{self.config.IV_LOW_PERCENTILE}), "
                f"DTE: {dte}. "
                f"Action based on {'High IV/Decay' if action == 'SELL_OPTION' else 'Direction/Low IV'}."
            )
        }

# --- MOCK SIMULATION (How to run in production) ---

if __name__ == "__main__":
    # Initialize Engine
    engine = OptionExecutionEngine()

    # ---------------------------------------------------------
    # SCENARIO 1: High Conviction Bull, Low IV (Ideal Buy)
    # ---------------------------------------------------------
    print("--- SCENARIO 1: Strong Bull, Cheap Options ---")
    layer1_input = {
        "direction": "BULLISH",
        "probability": 0.88
    }
    
    # Mocking Upstox Live Data
    market_context = {
        "spot_price": 24120.0,
        "expiry_date": "2024-03-28", # Assume today is 2024-03-20 (8 DTE)
        "current_iv": 12.5,
        "iv_history": [13.0, 14.5, 12.0, 15.0, 16.0, 11.5, 18.0, 12.0] # Mock 12.5 is low relative to this
    }

    decision = engine.execute_logic(layer1_input, market_context)
    print(json.dumps(decision, indent=2))


    # ---------------------------------------------------------
    # SCENARIO 2: Bearish, High IV, Near Expiry (Ideal Sell)
    # ---------------------------------------------------------
    print("\n--- SCENARIO 2: Bear, High Volatility, Expiry Soon ---")
    layer1_input_2 = {
        "direction": "BEARISH",
        "probability": 0.75
    }

    # Assume today is near expiry
    # Current IV 22.0 is high relative to history
    market_context_2 = {
        "spot_price": 24120.0,
        "expiry_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"), 
        "current_iv": 22.0,
        "iv_history": [13.0, 14.0, 15.0, 14.0, 12.0, 13.5] 
    }

    decision_2 = engine.execute_logic(layer1_input_2, market_context_2)
    print(json.dumps(decision_2, indent=2))

    # ---------------------------------------------------------
    # SCENARIO 3: Weak Signal (Gatekeeper Check)
    # ---------------------------------------------------------
    print("\n--- SCENARIO 3: Weak Signal ---")
    layer1_input_3 = {
        "direction": "BULLISH",
        "probability": 0.52 # Below 0.60 Threshold
    }
    
    decision_3 = engine.execute_logic(layer1_input_3, market_context) # Same market context
    print(json.dumps(decision_3, indent=2))