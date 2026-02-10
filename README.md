# ORION-NSE  
### Hybrid Algorithmic Trading System for NIFTY 50 Options

ORION-NSE is a **production-grade algorithmic trading system** designed for **NIFTY 50 options**.  
It follows a **3-layer architecture** that cleanly separates:

- **Prediction (Machine Learning)**
- **Strategy (Options Logic)**
- **Execution (Live Data & Risk Management)**

This separation improves robustness, prevents logic leakage, and enables safer deployment.

---

## 🏗 System Architecture

ORION-NSE operates on a **3-Layer “Brain → Context → Hands” protocol**.

| Layer | Component | Responsibility | Status |
|------|-----------|----------------|--------|
| **L1** | The Brain (XGBoost) | Predicts direction & probability (0–1) using Price, OI, and volatility features | ✅ Active |
| **L2** | The Context (`Logic.py`) | Decides *what strategy to use* (Buy Option vs Sell Spread) using IV percentile & expiry | ✅ Active |
| **L3** | The Hands (`app.py`) | Fetches live data, calculates **real ATM IV**, enforces risk rules, executes logic | ✅ Active |

---

## 📂 Project Structure

ORION-NSE/
│
├── data/
│ ├── NSE.json # Instrument keys (Upstox)
│ ├── Nifty_Hybrid_60days.csv # Raw historical data
│ └── Nifty_ML_Ready.csv # Processed ML features
│
├── Executions/
│ ├── init.py
│ └── Logic.py # Layer-2: Options strategy engine
│
├── Mode Training/
│ ├── models/
│ │ └── nifty_hybrid_model.pkl # Trained XGBoost model
│ ├── Train.py # Walk-forward training & validation
│ └── Refine_Sequential.py # Sequential labeling & diagnostics
│
├── app.py # Layer-3: Live orchestrator (ENTRY POINT)
├── fetch_and_process.py # Data pipeline (Upstox → CSV)
└── requirements.txt # Python dependencies


---

## 🚀 Installation & Setup

### 1️⃣ Prerequisites
- Python **3.9+**
- Upstox API account:
  - API Key
  - Secret
  - Redirect URI
  - Access Token (valid for 24 hours)

---

### 2️⃣ Install Dependencies

```bash
pip install upstox-python-sdk pandas numpy xgboost scikit-learn joblib py_vollib
3️⃣ Environment Configuration
Set your Upstox access token as an environment variable.

Windows (PowerShell)

$env:UPSTOX_ACCESS_TOKEN="your_access_token_here"
Linux / macOS

export UPSTOX_ACCESS_TOKEN="your_access_token_here"
⚔️ How to Run
🔁 Step 1: Update Data & Retrain (Daily / Weekend)
Fetch the latest data and retrain the model to adapt to new market regimes.

python fetch_and_process.py
python "Mode Training/Train.py"
▶️ Step 2: Live Trading (Market Hours)
Run the live orchestrator.
It evaluates the market every 15 minutes:

XX:00

XX:15

XX:30

XX:45

python app.py
Example Console Output
Fetching live market snapshot...
Real ATM IV: 14.5% (vs VIX: 12.0%)
L1 Prediction: Prob Buy = 0.72
L2 Decision: EXECUTE
Strategy: BULL_PUT_SPREAD
Default mode:

PAPER_TRADING = True

🛡️ Safety Protocols (Hard-Coded)
✅ Spread Enforcement
The system never sells naked options

Any sell signal is automatically converted into a credit spread

Example: Sell PE + Buy lower PE

✅ IV Integrity
Uses real ATM option IV

IV is computed via Black–Scholes (py_vollib)

India VIX is not used for trading decisions

✅ Kill Switch
MAX_TRADES_PER_DAY = 1
Prevents overtrading and execution loops.

✅ Model Gatekeeper
Trades are rejected if:

Model Probability < 60%
📊 Strategy Logic (Layer-2)
The system dynamically selects the strategy based on IV percentile and market context.

Scenario	Market Condition	Action	Strategy
A	Low IV (< 30%)	Buy Premium	Long Call / Long Put
B	High IV (> 70%)	Sell Premium	Credit Spread
C	Mid IV	Directional	Standard Buy
D	Neutral / Low Confidence	No Trade	Skip
⚠️ Risk Disclaimer
Alpha Status: This system is under active development.

Paper Trade First:

Do NOT set PAPER_TRADING = False until:

At least 20+ verified paper trades

Execution behavior matches expected logic

Market Risk:

Algorithmic trading involves significant financial risk.

API Dependency:

Depends on Upstox API availability and data accuracy.

No Liability:

The authors are not responsible for financial losses.

Final Note
This system is selective by design.
If it outputs NO TRADE, that is the signal.

Low frequency. High discipline.


---

If you want, next I can:
- add **badges** (Python version, status, license)
- split this into `docs/` pages
- or write a **Quick Start** section for new users