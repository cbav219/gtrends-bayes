# Usage — gtrends-bayes v4 forecasts

How to call the model, what each input means, and how to read the output.

## Supported horizons

| Code | Business days | When to use |
|---|---|---|
| `1d` | 1 | Live monitoring (requires daily cadence model) |
| `1w` | 5 | Default cadence; weekly PM reports |
| `2w` | 10 | Mid-month rebalancing review |
| `1m` | 21 | Monthly outlook |
| `1q` | 63 | Quarterly PM committee |

Longer horizons (`6m` = 126 BD, `1y` = 252 BD) are technically callable but
have small-N caveats — only ~18 non-overlapping yearly targets in the
training window. Don't lead with them.

## The two targets

| Frozen pickle | Underlying | Underlying ICE BofA series (for context) |
|---|---|---|
| `model/HY_v4.pkl` | HYG ETF (iShares iBoxx HY Corporate Bond) | `BAMLH0A0HYM2` |
| `model/IG_v4.pkl` | LQD ETF (iShares iBoxx IG Corporate Bond) | `BAMLC0A0CM` |

v4 uses ETF proxies because UChicago WRDS doesn't carry the aggregate ICE
BofA OAS index series and FRED's variant only goes back to 2023-05-02
(too short for the 2008-anchored training window).

> **Note — proxy quality.** Empirical correlation of weekly ETF log-returns
> against weekly ΔOAS (2023-05 → 2026-05, ~156 weeks):
>
> | Target | Pearson(ETF dlog, ΔOAS) | Notes |
> |---|---:|---|
> | HY (HYG ↔ HY OAS) | **−0.69** | Defensible — HYG is dominated by credit spread |
> | IG (LQD ↔ IG OAS) | **−0.24** | Weak — LQD has duration noise the Trends predictors don't capture |
>
> Read IG forecasts with extra grain-of-salt. If a longer-history OAS feed
> becomes available (Bloomberg, paid Nasdaq Data Link), the project's
> `IMPLEMENTATION_PLAN_v3.md` documents the retrain branch that would
> bypass the proxy entirely.

## The `as_of` parameter

`as_of` is the "decision day" — the date you're standing on when making
the forecast. The model uses:

- `y_history` up to `as_of` (inclusive).
- The most recent `x_latest` row up to `as_of` minus a publication lag
  (3 business days for daily Trends, 1 week for weekly Trends).

Use it for:
- **Live forecasting**: pass `as_of = today`.
- **Replay / backtest**: pass historical dates to see what the model
  would have said at each decision point.

## Interpreting the output

```python
{
    "target": "HY",
    "target_transform": "levels",
    "as_of": "2026-05-15",
    "horizon": "1m",
    "horizon_bd": 21,
    "n_draws": 1000,
    "conformal_alpha": 1.43,

    "median": 78.42,           # forecast in transform-space (here = levels)
    "q05": 72.10,              # 90% credible band, lower
    "q95": 84.71,              # 90% credible band, upper

    "level_median": 78.42,     # same as median for transform="levels" models
    "level_band": (72.10, 84.71),

    "path_median": [...],      # length-21 horizon path
    "path_q05":    [...],
    "path_q95":    [...],
    "level_path_median": [...],
    "level_path_q05":    [...],
    "level_path_q95":    [...],
}
```

For transform=`levels` models (the v4 ETF default), `median` and
`level_median` are the same number. For transform=`log_diff` models (v3
Phase D retraining output), `median` is a cumulative log-return at the
horizon — use `level_median` for the PM-facing number.

The 90% band has been **conformal-recalibrated**: the multiplier α was
learned on a held-out validation slice so the raw 90% MCMC band achieves
its nominal coverage. The α value is in `conformal_alpha` for traceability
(typically 1.3–1.6 for these models).

## Worked example — CLI (real data)

After unpacking both the model bundle and the data sideband:

```bash
$ python scripts/example_forecast.py --real
v4 inference example (real data)
========================================================================

HY  [transform=levels, cadence=weekly, α=0.948]
------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 79.71
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.40      -3.70      +5.98         81.12      76.01      85.69
2w          +1.11      -5.41      +7.12         80.82      74.31      86.84
1m          +0.85      -7.29      +9.16         80.56      72.42      88.87
1q          +1.13     -13.75     +14.67         80.84      65.97      94.38

IG  [transform=levels, cadence=weekly, α=1.222]
------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 108.31
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.59      -6.51      +8.73        109.89      97.33     112.57
1m          +0.94     -14.18     +16.65        104.78      89.66     120.49
1q          +2.02     -27.66     +31.20        105.86      76.18     135.04
```

Read each row as: at this horizon, the median forecast is `level_median`
with a 90% credible band `[level 5%, level 95%]`. Band width grows ~√h
with horizon — exactly what you'd expect from near-random-walk dynamics.

## Worked example — Jupyter-cell API

In a notebook cell on the JupyterLab VM:

```python
from gtrends_bayes.inference import load_model, forecast
import pandas as pd

# 1. Load the frozen model for HY (or IG).
model = load_model("model/HY_v4.pkl")

# 2. Load the data sideband.
y = pd.read_csv("data/HY_history.csv", parse_dates=[0], index_col=0).iloc[:, 0]
x = pd.read_parquet("data/trends.parquet")

# 3. Forecast 1 month ahead from the latest observed date.
out = forecast(model, "1m", y.index.max(), y, x, n_draws=1000, seed=42)

print(f"HY 1m forecast")
print(f"  as of:        {out['as_of'][:10]}")
print(f"  median level: ${out['level_median']:.2f}")
print(f"  90% band:     [${out['level_band'][0]:.2f}, ${out['level_band'][1]:.2f}]")
print(f"  conformal α:  {out['conformal_alpha']:.3f}")
```

`out` also includes full-path arrays (`level_path_median`, `level_path_q05`,
`level_path_q95`, each length `horizon_bd`) for fan-chart plotting:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
days = range(1, out["horizon_bd"] + 1)
ax.fill_between(days, out["level_path_q05"], out["level_path_q95"], alpha=0.25, label="90% band")
ax.plot(days, out["level_path_median"], lw=2, label="median")
ax.axhline(float(y.iloc[-1]), color="k", ls=":", label="last observed")
ax.set_xlabel("business days ahead"); ax.set_ylabel("HY ETF price (USD)")
ax.set_title(f"HY forecast — {out['horizon']} horizon as of {out['as_of'][:10]}")
ax.legend(); plt.tight_layout()
```

## When the forecast looks wrong

| Symptom | Likely cause |
|---|---|
| Median far from last observed | Coefficient drift since training; refit upstream |
| Band exploding (factor of 1000) | Transform mismatch — confirm `target_transform` |
| All draws identical | Wrong predictor column set in x_latest |
| Stale forecast (band centered on old level) | y_history doesn't extend to as_of - 1 |

## What this model is and isn't

**Is:** a Trends-driven *supplement* to AR baselines, calibrated so the
90% band has correct coverage, exposing both a point forecast and a
posterior-spread regime indicator (the Trends Risk Index, published
separately in `data/processed/risk_index/`).

**Isn't:** an RMSE-beating point forecaster. The model trades weekly RMSE
for directional signal. From the v3 horizon sweep (40 rows):

| Target | Horizon | BSTS hit rate | Naive RW hit rate |
|---|---|---:|---:|
| HY | 1q | **0.609** | 0.302 |
| HY | 6m | **0.619** | 0.198 |
| IG | 1m | **0.554** | 0.505 |
| IG | 6m | **0.515** | 0.446 |

BSTS wins meaningfully at long horizons (3× signal vs naive at HY 6m).
Use it as a *direction-and-regime* tool, not a price oracle. PMs reading
the median as a literal predicted price will be disappointed; reading it
as "the model leans modestly cheaper here, with wide uncertainty" is the
right framing.
