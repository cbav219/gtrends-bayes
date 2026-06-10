# Data sideband — file format contract

The v5 bundle is the **model + inference layer**. The **data** (price
history + preprocessed Trends predictors) ships separately as multiple
smaller emails so each attachment stays under the JupyterLab inbox cap
(< 20 MB per email).

This document specifies what files belong in `data/` and the format
each one must satisfy. `scripts/verify_data.py` validates this contract.

## Expected layout

```
gtrends-bayes-v5/
└── data/
    ├── HY_history.csv         # y for HY (HYG total-return USD)
    ├── IG_history.csv         # y for IG (LQD total-return USD)
    ├── trends.parquet         # X — all 43 preprocessed predictors
    └── README.md              # this file
```

## File formats

### `HY_history.csv` / `IG_history.csv`

Two-column CSV. First column = date, second column = level. Header row
required (column names can be anything; the loader uses positions).

```
date,close
2024-01-05,78.42
2024-01-12,79.10
2024-01-19,77.86
...
```

Rules:
- ISO-8601 dates (`YYYY-MM-DD`).
- One row per Sunday-aligned weekly bar (or per business day if daily-
  cadence model is shipped — check `model.preprocessing.cadence`).
- Values in the same units BSTS was trained on: HYG / LQD total-return
  closing prices in USD.
- At minimum 50 rows. AR(4) backbone needs the last 4 observations of
  history; downstream coverage diagnostics want at least a year of
  history if possible.

### `trends.parquet`

Wide-format pandas DataFrame, indexed by date, with one column per
predictor.

Rules:
- Column names exactly match `model.bsts_posterior.X_columns`. The
  full list is printed by `scripts/verify_data.py`; here are the first
  few: `Agriculture & Forestry`, `Apparel`, `Auto Financing`, …,
  `vix`, `ust10y`, `ust2y10y_slope`. Total 43 columns.
- Index is dates (business-day for daily-cadence models, weekly Sunday
  for weekly-cadence).
- Values are **already preprocessed** — multi-sample averaged, drift-
  removed (HP filter), YoY-differenced for categorical predictors. The
  inference module does NOT re-run preprocessing; it only realigns
  column order. If you ship raw SVI here, the forecasts will be
  wildly miscalibrated.

The upstream `scripts/run_preprocessing.py` produces a file with the
correct shape. The data-sideband emails contain that output, possibly
split into multiple parquet files if size exceeds the email cap —
concatenate them client-side along the date axis before placing as
`trends.parquet`.

## Stale-data behavior

`verify_data.py` warns (but doesn't fail) if the latest data is older
than 30 days. The inference module will still produce forecasts, but
the `as_of` date will be artificially old — fine for backtesting,
misleading for "live" calls. To live-forecast, ensure both `*_history.csv`
and `trends.parquet` extend within the last ~7 business days.

## Sidebanding plan (for the sender, Cesare)

If the total `trends.parquet` exceeds ~10 MB, split by predictor *group*
(`labor`, `credit_lending`, `consumption`, …) per the upstream
`config/predictors.yaml`. Each group fits in one email. Reassemble on
the VM with a simple `pd.concat(axis=1)` — the unpack helper in
`scripts/verify_data.py` can do this if you provide a manifest naming
which files to merge. (Helper not yet wired; if needed, ship a single
parquet for now and split later.)
