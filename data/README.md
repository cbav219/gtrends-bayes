# `data/` — not included in this repository

The datasets and frozen model artifacts this project was built on are **not
shipped here**. They are proprietary and were produced for **Neuberger Berman**
as part of the University of Chicago Project Lab, so they are intentionally kept
out of the public repository.

What normally lives under `data/` (and how the pipeline expects it):

| Path | Contents |
|---|---|
| `data/raw/` | Weekly Google Trends cache + ETF/FRED target pulls |
| `data/raw_daily/` | Daily Trends cache (41-predictor universe) |
| `data/processed/` | Preprocessed Trends matrix, BSTS posteriors, backtest CSVs, recalibration JSON, Risk Index outputs |
| `data/csv/` | User-supplied FRED OAS reference series (2023-05+) |

Representative shapes (from the v5 deliverable):

- `*_history.csv` — 957 weekly ETF closes, 2008-01 → 2026-05 (HYG for HY, LQD for IG)
- `trends.parquet` — 957 × 43 preprocessed Trends matrix (41 predictors + 2 controls)
- Frozen models — `HY_v5.pkl` / `IG_v5.pkl` (pure-Python inference artifacts)

The full data schema is documented in [`docs/v5/data_README.md`](../docs/v5/data_README.md).

## Running without the data

The repository still runs end-to-end on **synthetic inputs** — no data or model
pickle required:

```bash
python scripts/example_forecast.py        # synthetic smoke test
```

To reproduce the real pipeline you supply your own Google Trends pulls and a free
[FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html); see the top-level
`README.md` and `docs/v5/USAGE.md`.
