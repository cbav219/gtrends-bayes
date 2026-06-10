# Usage — gtrends-bayes v5 forecasts

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
| `model/HY_v5.pkl` | HYG ETF (iShares iBoxx HY Corporate Bond) | `BAMLH0A0HYM2` |
| `model/IG_v5.pkl` | LQD ETF (iShares iBoxx IG Corporate Bond) | `BAMLC0A0CM` |

v5 uses ETF proxies because UChicago WRDS doesn't carry the aggregate ICE
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

For transform=`levels` models (the v5 ETF default), `median` and
`level_median` are the same number. For transform=`log_diff` models (v3
Phase D retraining output), `median` is a cumulative log-return at the
horizon — use `level_median` for the PM-facing number.

The 90% band has been **conformal-recalibrated**: the multiplier α was
learned on a held-out validation slice so the raw 90% MCMC band achieves
its nominal coverage. The α value is in `conformal_alpha` for traceability
(typically 1.3–1.6 for these models).

## Reading OAS-implied output (ETF models only)

For the ETF targets (`HY`, `IG`), the frozen pickle also carries an
empirical ETF↔OAS regression baked in at freeze time. When `forecast()`
runs on these models it emits **extra fields** that re-express the
ETF-price forecast in basis-points OAS, anchored on the latest observed
ICE BAML OAS level:

```python
{
    # ... all the ETF fields above, plus:
    "oas_implied_median": 261.6,            # bps
    "oas_implied_band":   (89.9, 448.0),    # bps, ordered low → high
    "oas_implied_path_median":   [...],      # length-horizon_bd path
    "oas_implied_path_band_lo":  [...],
    "oas_implied_path_band_hi":  [...],
    "oas_overlay_meta": {
        "slope_bps_per_dlog": -1738.74,
        "pearson": -0.691,                   # overlap-window correlation
        "n_overlap_weeks": 154,              # FRED data, 2023-05+
        "last_oas_bps": 280.0,
        "last_oas_date": "2026-05-17",
        "proxy_quality_label": "defensible", # or "weak" for IG
    },
}
```

Math: `oas_implied = last_oas_bps + slope · ln(level_forecast / last_level)`.
Slope is the OLS regression of weekly ΔOAS-bps on weekly ETF-Δlog over the
~3-year overlap window (Pearson · σ_oas / σ_etf).

> **Proxy-quality caveat.** HY is defensible (Pearson −0.69 over n=154
> weeks). IG is weak (Pearson −0.24) because LQD has duration noise the
> Trends predictors don't directly model. Read IG OAS-implied numbers
> with extra grain-of-salt.

## Reading OAS-direct output (HY_OAS / IG_OAS models)

For the v5.1 auxiliary `HY_OAS_v5.pkl` / `IG_OAS_v5.pkl` pickles
(direct BSTS fits on the FRED OAS history, 156 weekly bars from 2023-05):

- `target` is `"HY_OAS"` / `"IG_OAS"` (not `"HY"` / `"IG"`).
- `level_median` / `level_band` are in **basis points** directly — no
  translation needed.
- The pickle's `history_file` field is `HY_OAS_history.csv` /
  `IG_OAS_history.csv`. The included `_load_real_y` helper in
  `scripts/example_forecast.py` reads each pickle's `history_file`
  automatically so you don't have to wire the right CSV by hand.
- No `oas_overlay_translation` block (no translation is required — the
  model already speaks bps).

The OAS-direct fit is **small-N** (n≈100 backtest points). Conformal
recalibration lands in [0.75, 0.85] in-sample on all 8 cells, but the
out-of-sample band is wider because of the short training history. Use
this as a **second opinion** alongside the OAS-implied translation —
when they disagree, both are useful PM information about regime
uncertainty.

## Worked example — CLI (real data)

After unpacking both the model bundle and the data sideband:

```bash
$ python scripts/example_forecast.py --real
v5 inference example (real data)
========================================================================

HY  [transform=levels, cadence=weekly, α=0.954]
------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 79.71
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.40      -3.73      +6.01         81.12      75.98      85.72
2w          +1.11      -5.45      +7.16         80.82      74.27      86.87
1m          +0.85      -7.34      +9.21         80.56      72.37      88.92
1q          +1.13     -13.84     +14.75         80.84      65.87      94.47

IG  [transform=levels, cadence=weekly, α=1.206]
------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 108.31
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.59      -7.68     +10.52        109.89     100.63     118.83
2w          +1.44     -10.49     +12.47        109.74      97.82     120.78
1m          +0.65     -14.06     +17.81        108.96      94.25     126.12
1q          +0.52     -29.04     +30.16        108.82      79.27     138.47
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
model = load_model("model/HY_v5.pkl")

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
