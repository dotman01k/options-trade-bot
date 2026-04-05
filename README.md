# Options Trade Bot Dashboard

This project is a lightweight options scanner and dashboard built with Streamlit.
It ranks option contracts using a practical score based on:

1. liquidity
2. bid ask spread quality
3. proximity to spot
4. affordability
5. implied volatility
6. delta fit
7. simple trend bias from the underlying

## Important note

This is a research and paper trading tool.
It does **not** guarantee the "best" trade, and it should not place live trades without adding your own broker integration, position sizing rules, stop logic, and compliance checks.

## Features

- Pulls price history and option chains with `yfinance`
- Computes a simple directional bias using SMA, MACD, and RSI
- Filters contracts by days to expiration, volume, open interest, and spread
- Scores and ranks contracts for calls, puts, or both
- Shows a dashboard with:
  - top option candidates
  - score bubble chart
  - underlying price trend
  - RSI and MACD charts
  - CSV export

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## How the score works

The ranking is a weighted score from 0 to 100. Higher is better for the current filter set.

- 24% liquidity
- 20% spread quality
- 14% strike proximity to spot
- 12% affordability
- 10% implied volatility favorability
- 10% delta fit near a practical swing-trade zone
- 10% agreement with the current price trend bias

## Files

- `app.py` — Streamlit dashboard
- `data.py` — market data and technical indicators
- `scanner.py` — filters, scoring, and ranking logic
- `requirements.txt` — Python dependencies

## Next upgrades you can add

- broker integration for paper trading or live alerts
- earnings calendar and macro event filters
- IV rank and IV percentile from a richer data source
- strategy support for verticals, calendars, and iron condors
- alerting by email, SMS, or Discord
- backtesting and trade journaling
