# ORION-NSE

A probability-driven intraday option signal engine for the Indian NSE market.

ORION-NSE uses a machine learning model trained on historical NIFTY price action
to filter high-probability option trades. The system is intentionally conservative
and designed to avoid overtrading.

---

## Key Features

- Machine-learning–based trade filtering
- Supports two locked strategies per session:
  - OPTION BUYING
  - OPTION SELLING
- Live market data via Upstox API
- 5-minute candle analysis
- Strict risk controls:
  - Max trades per day
  - Cooldown between signals
  - Time-based cutoff
- Signal-only system (no auto order placement)

---

## Project Structure

.
├── Training.py # Model training pipeline
├── Trading.py # Live signal engine (Upstox)
├── README.md
├── .gitignore


---

## How It Works (High Level)

1. Train a classification model on historical NIFTY data
2. Save the trained model, scaler, and feature list
3. Run the live engine before market hours
4. Select strategy (BUYING or SELLING)
5. System generates structured trade signals during market hours

The model prioritizes **precision over recall**.
Missing trades is acceptable. Bad trades are not.

---

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- joblib
- requests
- Upstox API access

---

## Disclaimer

This project is for research and educational purposes only.

Trading and derivatives involve significant risk.
The author is not responsible for financial losses.