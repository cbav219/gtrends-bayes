# Bootstrap — gtrends-bayes v5

One-page setup for the NB JupyterLab VM (or any Python 3.10+ environment).
Four commands and you have a working forecast.

## 1. Unpack the model bundle

You should have received **two emails**:

1. `gtrends-bayes-v5.tar.gz` — the frozen model + inference layer (~30 KB)
2. `gtrends-bayes-v5-data.zip` — the data sideband: HY/IG history + Trends
   parquet (~270 KB)

```bash
tar -xzf gtrends-bayes-v5.tar.gz
cd gtrends-bayes-v5
```

The directory now contains `model/`, `src/`, `scripts/`, and the markdown
docs.

## 2. Install Python deps

The VM kernel likely has most of these already (pandas, numpy, pyarrow).

```bash
pip install -r requirements.txt
```

No R install is needed — this bundle does not re-fit anything.

## 3. Unpack the data sideband into `data/`

```bash
unzip ../gtrends-bayes-v5-data.zip -d data/
```

This populates `data/` with `HY_history.csv`, `IG_history.csv`,
`trends.parquet`, `HY_OAS_history.csv`, `IG_OAS_history.csv`, and an
internal `README.md` documenting the formats.

Verify everything is in place:

```bash
python scripts/verify_data.py
```

**Expected output:**

```
✓ model: HY_v5.pkl (target=HY, transform=levels, cadence=weekly)
✓ model: IG_v5.pkl (target=IG, transform=levels, cadence=weekly)
✓ y: HY_history.csv — OK (957 rows, 2008-01-06 → 2026-05-03)
✓ y: IG_history.csv — OK (957 rows, 2008-01-06 → 2026-05-03)
✓ X: trends.parquet — OK (957 rows × 43 cols, 2007-12-30 → 2026-04-26)
Ready to run inference.
```

If any line starts with `✗`, the message tells you exactly what's wrong —
usually a missing file or a column-name mismatch between the data sideband
and the model bundle (confirm both came from the same `v5` send).

## 4. Run a forecast

The bundled one-pager iterates both targets across the PM horizon ladder:

```bash
python scripts/example_forecast.py --real
```

**Expected output (numbers will vary with as_of):**

```
v5 inference example (real data)
========================================================================

HY_OAS  [HY OAS-direct (bps, ICE BAML BAMLH0A0HYM2)]
        transform=levels, cadence=weekly, α=1.153
----------------------------------------------------------------------------------------
as_of: 2026-05-17    last observed: 280.00 bps
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +17.77   -106.59    +139.90        297.77     173.41     419.90
1q          +29.60   -220.93    +299.61        309.60      59.07     579.61

HY  [HYG ETF (USD) — proxy for HY OAS]
        transform=levels, cadence=weekly, α=0.954
----------------------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 79.71 USD
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.40      -3.73      +6.01         81.12      75.98      85.72
1m          +0.85      -7.34      +9.21         80.56      72.37      88.92
1q          +1.13     -13.84     +14.75         80.84      65.87      94.47

        OAS-implied (via ETF↔OAS regression): slope=-1738.7 bps/dlog, pearson=-0.69 (defensible), n=154 wk overlap
        last OAS anchor: 280 bps (2026-05-17)
horizon   Δ bps median   Δ bps low  Δ bps high  OAS median    OAS low   OAS high
1w              -30.35     -126.32      +83.32       249.7      153.7      363.3
1m              -18.42     -190.15     +167.98       261.6       89.9      448.0
1q              -24.41     -295.27     +331.52       255.6      -15.3      611.5

IG_OAS  [IG OAS-direct (bps, ICE BAML BAMLC0A0CM)]
        transform=levels, cadence=weekly, α=1.019
----------------------------------------------------------------------------------------
as_of: 2026-05-17    last observed: 75.00 bps
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w           +4.53    -20.07     +28.10         79.53      54.93     103.10
1q          +10.71    -41.32     +64.44         85.71      33.68     139.44

IG  [LQD ETF (USD) — proxy for IG OAS]
        transform=levels, cadence=weekly, α=1.206
----------------------------------------------------------------------------------------
as_of: 2026-05-03    last observed: 108.31 USD
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.59      -7.68     +10.52        109.89     100.63     118.83
1m          +0.65     -14.06     +17.81        108.96      94.25     126.12
1q          +0.52     -29.04     +30.16        108.82      79.27     138.47

        OAS-implied (via ETF↔OAS regression): slope=-91.9 bps/dlog, pearson=-0.24 (weak), n=154 wk overlap
        last OAS anchor: 75 bps (2026-05-17)
horizon   Δ bps median   Δ bps low  Δ bps high  OAS median    OAS low   OAS high
1w               -1.34       -8.52       +6.76        73.7       66.5       81.8
1m               -0.55      -14.00      +12.78        74.4       61.0       87.8
1q               -0.44      -22.59      +28.69        74.6       52.4      103.7
```

For one-off forecasts at a specific horizon and `as_of` date, use the CLI:

```bash
python -m gtrends_bayes.inference \
    --model-path model/HY_v5.pkl \
    --horizon 1m \
    --as-of 2026-05-15 \
    --y-data data/HY_history.csv \
    --x-data data/trends.parquet
```

Or from a Jupyter cell:

```python
from gtrends_bayes.inference import load_model, forecast
import pandas as pd

model = load_model("model/HY_v5.pkl")
y = pd.read_csv("data/HY_history.csv", parse_dates=[0], index_col=0).iloc[:, 0]
x = pd.read_parquet("data/trends.parquet")
out = forecast(model, "1m", pd.Timestamp("2026-05-15"), y, x)
print(out["level_median"], out["level_band"])
```

See `USAGE.md` for the full output schema, the proxy-quality caveat, and
the fan-chart plot recipe.

## Troubleshooting

- **`verify_data.py` says "missing X columns"** — the trends parquet was
  built from a different predictor set than the model was trained on.
  Confirm both attachments came from the same `v5` send.
- **`forecast` returns absurd numbers** — likely a transform mismatch.
  Models with `transform=levels` expect raw level y (e.g. HYG closing
  prices in USD); models with `transform=log_diff` expect levels too but
  compute log-returns internally. Don't pre-transform y yourself.
- **`forecast` is slow (> 5 s)** — `n_draws` defaults to 1000; drop to
  200 for live monitoring, or 50 for dev iteration.
