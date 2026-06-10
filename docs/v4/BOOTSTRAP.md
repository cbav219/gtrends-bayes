# Bootstrap — gtrends-bayes v4

One-page setup for the NB JupyterLab VM (or any Python 3.10+ environment).
Four commands and you have a working forecast.

## 1. Unpack the model bundle

You should have received **two emails**:

1. `gtrends-bayes-v4.tar.gz` — the frozen model + inference layer (~30 KB)
2. `gtrends-bayes-v4-data.zip` — the data sideband: HY/IG history + Trends
   parquet (~270 KB)

```bash
tar -xzf gtrends-bayes-v4.tar.gz
cd gtrends-bayes-v4
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
unzip ../gtrends-bayes-v4-data.zip -d data/
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
✓ model: HY_v4.pkl (target=HY, transform=levels, cadence=weekly)
✓ model: IG_v4.pkl (target=IG, transform=levels, cadence=weekly)
✓ y: HY_history.csv — OK (957 rows, 2008-01-06 → 2026-05-03)
✓ y: IG_history.csv — OK (957 rows, 2008-01-06 → 2026-05-03)
✓ X: trends.parquet — OK (957 rows × 43 cols, 2007-12-30 → 2026-04-26)
Ready to run inference.
```

If any line starts with `✗`, the message tells you exactly what's wrong —
usually a missing file or a column-name mismatch between the data sideband
and the model bundle (confirm both came from the same `v4` send).

## 4. Run a forecast

The bundled one-pager iterates both targets across the PM horizon ladder:

```bash
python scripts/example_forecast.py --real
```

**Expected output (numbers will vary with as_of):**

```
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
1m          +0.94     -14.18     +16.65        108.96      94.05     126.35
1q          +2.02     -27.66     +31.20        108.82      78.87     138.86
```

For one-off forecasts at a specific horizon and `as_of` date, use the CLI:

```bash
python -m gtrends_bayes.inference \
    --model-path model/HY_v4.pkl \
    --horizon 1m \
    --as-of 2026-05-15 \
    --y-data data/HY_history.csv \
    --x-data data/trends.parquet
```

Or from a Jupyter cell:

```python
from gtrends_bayes.inference import load_model, forecast
import pandas as pd

model = load_model("model/HY_v4.pkl")
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
  Confirm both attachments came from the same `v4` send.
- **`forecast` returns absurd numbers** — likely a transform mismatch.
  Models with `transform=levels` expect raw level y (e.g. HYG closing
  prices in USD); models with `transform=log_diff` expect levels too but
  compute log-returns internally. Don't pre-transform y yourself.
- **`forecast` is slow (> 5 s)** — `n_draws` defaults to 1000; drop to
  200 for live monitoring, or 50 for dev iteration.
