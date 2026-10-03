# Forward Sentiment Scanner

Streamlit app that pulls forward-looking analyst data from yfinance for your NSE watchlist
and scores rows G1–G4 of the Forward Sentiment Matrix.

## Run

```bash
cd forward_scanner
pip install -r requirements.txt
streamlit run app.py
```

Click **Fetch fresh data** in the sidebar. Results are saved to `snapshots/forward_YYYY-MM-DD.csv`
and reloaded automatically next time, so you only fetch once a day.

## Files

| File | Purpose |
|---|---|
| `app.py` | The app |
| `stocks.csv` | Your 288-row TradingView universe (287 unique stocks; columns section, exchange, symbol). Any CSV with a `symbol` column also works. `NAM_INDIA`→`NAM-INDIA`, `BAJAJ_AUTO`→`BAJAJ-AUTO`, BSE rows → `.BO` |
| `snapshots/` | Daily fetches; keep them — they build the history for tracking target-price changes later |
| `screener_queries.md` | Paste-ready Screener.in screens for the fundamentals pillars |

## What it scores

| ID | Metric | Source |
|---|---|---|
| G1 | Mean target ÷ price − 1 | `info.targetMeanPrice` |
| G2 | Forward PE ÷ TTM PE | price ÷ `forwardEps`, price ÷ `trailingEps` |
| G3 | Forward PE ÷ next-FY EPS growth | `eps_trend` (+1y vs 0y) |
| G4 | Next-FY EPS consensus change over 90 days | `eps_trend` (current vs 90daysAgo) |

Two scoring modes (sidebar):

* **Relative (default)** — each metric is ranked within your list: best third +1, middle 0, worst third −1.
* **Absolute** — the matrix's fixed thresholds. In testing on your 287 stocks this rated 191 of 238 covered
  stocks Positive or Strong positive, because analyst numbers are optimistic across the board
  (median target upside was 21%). Use it only for comparison.

Weights match the matrix and can be changed in the sidebar.

## Export

Below the table: **Download Excel (.xlsx)** — the filtered table with formatted numbers, a red→green
score scale and +1/−1 shading per metric, plus a *Settings* sheet recording the data date, scoring mode,
minimum analysts and the weights/thresholds used. **Download CSV** gives the same rows unformatted.
Both work on Streamlit Community Cloud (the file goes straight to your browser).

## Data checks built in

* Stocks with fewer analysts than the minimum (default 3) get no forward score.
* If `forwardEps` and the `eps_trend` table disagree by more than 10%, G2–G4 are dropped
  and the row is flagged "EPS sources disagree" (seen for RELIANCE in testing).
* Symbols Yahoo doesn't recognise are listed under "Fetch errors".

## Known limits

* yfinance is unofficial; Yahoo can rate-limit or change fields. If fetches fail, lower
  "Parallel requests" to 1–2 and retry.
* Small caps usually have no analyst coverage → "No forward data".
* The analyst up/down revision counts from yfinance are internally inconsistent, so they
  are displayed but not scored.
* Thresholds and weights are untested. Not investment advice.
