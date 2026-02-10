import numpy as np
from datetime import datetime

class Config:
    # Nifty Specifics
    STRIKE_INTERVAL = 50 
    
    # Layer-1 Thresholds
    MODEL_CONFIDENCE_THRESHOLD = 0.60
    HIGH_CONFIDENCE_THRESHOLD = 0.85

    # IV Thresholds (Percentile)
    IV_LOW_PERCENTILE = 30
    IV_HIGH_PERCENTILE = 70

    # Expiry Thresholds
    NEAR_EXPIRY_DAYS = 2

class OptionExecutionEngine:
    def __init__(self):
        self.config = Config()

    def _get_days_to_expiry(self, expiry_date_str: str) -> int:
        expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        return max(0, (expiry - datetime.now()).days + 1)

    def _calculate_iv_percentile(self, current_iv, iv_history):
        if not iv_history: return 50.0
        return (np.sum(np.array(iv_history) < current_iv) / len(iv_history)) * 100

    def _decide_strategy_action(self, iv_percentile, dte, direction):
        if direction == "NEUTRAL": return "NO_TRADE"
        
        # High IV + Near Expiry = SELL PREMIUM (Credit Spreads)
        if iv_percentile > self.config.IV_HIGH_PERCENTILE and dte <= self.config.NEAR_EXPIRY_DAYS:
            return "SELL_SPREAD"
            
        # Low/Normal IV = BUY PREMIUM (Debit Trades)
        return "BUY_OPTION"

    def _construct_legs(self, spot, direction, action, prob):
        """
        Constructs the specific legs (Main Leg + Hedge Leg).
        """
        atm_strike = round(spot / self.config.STRIKE_INTERVAL) * self.config.STRIKE_INTERVAL
        legs = []
        strategy_name = ""

        # --- SCENARIO A: BUY OPTION (Long Directional) ---
        if action == "BUY_OPTION":
            # Logic: Buy ATM or slightly OTM if confident
            strike_step = 1 if prob > self.config.HIGH_CONFIDENCE_THRESHOLD else 0
            
            if direction == "BULLISH":
                strike = atm_strike + (strike_step * self.config.STRIKE_INTERVAL)
                legs.append({"side": "BUY", "type": "CE", "strike": int(strike)})
                strategy_name = "LONG_CALL"
                
            elif direction == "BEARISH":
                strike = atm_strike - (strike_step * self.config.STRIKE_INTERVAL)
                legs.append({"side": "BUY", "type": "PE", "strike": int(strike)})
                strategy_name = "LONG_PUT"

        # --- SCENARIO B: SELL SPREAD (Credit Strategy) ---
        # SAFETY CRITICAL: NEVER SELL NAKED
        elif action == "SELL_SPREAD":
            # Logic: Sell OTM (Main) + Buy Further OTM (Hedge)
            # Spread Width: 2 Strikes (100 points) for Nifty
            SPREAD_WIDTH = 2 * self.config.STRIKE_INTERVAL 
            
            if direction == "BULLISH":
                # Strategy: BULL PUT SPREAD (Sell PE, Buy Lower PE)
                # Sell 1 strike OTM Put
                short_strike = atm_strike - self.config.STRIKE_INTERVAL
                long_strike  = short_strike - SPREAD_WIDTH # Hedge
                
                legs.append({"side": "SELL", "type": "PE", "strike": int(short_strike)})
                legs.append({"side": "BUY",  "type": "PE", "strike": int(long_strike)})
                strategy_name = "BULL_PUT_SPREAD"
                
            elif direction == "BEARISH":
                # Strategy: BEAR CALL SPREAD (Sell CE, Buy Higher CE)
                # Sell 1 strike OTM Call
                short_strike = atm_strike + self.config.STRIKE_INTERVAL
                long_strike  = short_strike + SPREAD_WIDTH # Hedge
                
                legs.append({"side": "SELL", "type": "CE", "strike": int(short_strike)})
                legs.append({"side": "BUY",  "type": "CE", "strike": int(long_strike)})
                strategy_name = "BEAR_CALL_SPREAD"

        return strategy_name, legs

    def execute_logic(self, layer1_output: dict, market_data: dict) -> dict:
        # 1. Gatekeeper
        prob = layer1_output.get("probability", 0.0)
        direction = layer1_output.get("direction")
        
        if prob < self.config.MODEL_CONFIDENCE_THRESHOLD:
            return {"decision": "NO_TRADE", "reason": f"Low Prob ({prob:.2f})"}
        if direction == "NEUTRAL":
            return {"decision": "NO_TRADE", "reason": "Neutral Direction"}

        # 2. Context
        spot = market_data['spot_price']
        iv_pct = self._calculate_iv_percentile(market_data['current_iv'], market_data['iv_history'])
        dte = self._get_days_to_expiry(market_data['expiry_date'])

        # 3. Decision
        action = self._decide_strategy_action(iv_pct, dte, direction)
        
        if action == "NO_TRADE":
            return {"decision": "NO_TRADE", "reason": "Market Context Mismatch"}

        # 4. Construct Legs (Safety First)
        strategy, legs = self._construct_legs(spot, direction, action, prob)

        return {
            "decision": "EXECUTE",
            "strategy": strategy,
            "action_type": action, # BUY_OPTION or SELL_SPREAD
            "legs": legs,
            "context": {
                "iv_percentile": round(iv_pct, 1),
                "dte": dte,
                "spot": spot
            }
        }