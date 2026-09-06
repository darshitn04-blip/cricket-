# CricoNomics | IPL Auction Value Analyzer

An interview-ready sports analytics project that asks a sharp business question: **are IPL players over- or underpriced relative to their on-field contribution?**

## Highlights

- Calculates role-normalised performance value from batting, bowling and availability signals.
- Uses a scikit-learn Random Forest to estimate a fair auction price and flag value picks or premiums.
- Clusters teams by auction behaviour: star-led, balanced, or value-led.
- Includes an interactive Streamlit dashboard, a price scenario lab, CSV export and SQLite views for Tableau/Power BI.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Real-data path

The default player table is intentionally labelled **illustrative** so the app starts immediately without making false claims. For a production-quality version:

1. Download ball-by-ball IPL files from [Cricsheet](https://cricsheet.org/downloads/).
2. Aggregate player-season runs, wickets, strike rate, economy and appearances.
3. Add audited auction prices, then upload a CSV containing:

```text
player,team,role,auction_price_cr,runs,wickets,strike_rate,economy,matches
```

4. Validate predictions against future auction seasons, not random rows from the same season.


