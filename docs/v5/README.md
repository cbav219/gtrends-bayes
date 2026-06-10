# gtrends-bayes v5 — frozen forecast bundle

This bundle lets you run weekly / monthly / quarterly **HY** and **IG**
corporate-bond ETF forecasts on demand using a frozen Bayesian Structural
Time Series model trained on a 41-predictor Google Trends library.

**What it forecasts:** weekly HYG (iShares HY) and LQD (iShares IG) ETF
closing prices, used as proxies for HY OAS and IG OAS spreads. See
`USAGE.md` §"The two targets" for the proxy-quality caveat.

**4-command setup (see `BOOTSTRAP.md`):**

```bash
tar -xzf gtrends-bayes-v5.tar.gz && cd gtrends-bayes-v5
unzip ../gtrends-bayes-v5-data.zip -d data/
pip install -r requirements.txt
python scripts/example_forecast.py --real
```

## What's inside

| File | Purpose |
|---|---|
| `model/HY_v5.pkl` | Frozen posterior + AR backbone + preprocessing state for HY (HYG proxy) |
| `model/IG_v5.pkl` | Same for IG (LQD proxy) |
| `src/gtrends_bayes/inference/` | Pure-Python load + forecast module (no R, no MCMC) |
| `scripts/verify_data.py` | Sanity-check the data-sideband files before forecasting |
| `scripts/example_forecast.py` | One-pager showing PM-facing horizon ladder output |
| `requirements.txt` | Minimum runtime deps (pandas, numpy, pyarrow, pyyaml) |
| `BOOTSTRAP.md` | One-page VM setup walkthrough |
| `USAGE.md` | Two-page forecast walkthrough + interpretation |
| `data/README.md` | Spec for the separately-shipped data sideband |

## What it does NOT do

- **Re-fit the model.** Posteriors are frozen. v5 is for *using* the model;
  retraining lives in the upstream `gtrends-bayes` repo (v3 pipeline).
- **Pull data from Google.** Trends ingest is a 32-hour overnight job
  upstream; the prepared parquet ships separately as the data sideband.
- **Need R / rpy2.** All inference is pure Python.

## What this bundle is for

The PM workflow is:
1. Drop the data-sideband files into `data/` (see `data/README.md`).
2. Run `python scripts/verify_data.py` to confirm the layout is right.
3. Call `python -m gtrends_bayes.inference --target HY --horizon 1m ...`
   (or the `forecast()` function from a notebook cell) to get a point
   forecast + 90% credible band at any of the five supported horizons
   (`1d`, `1w`, `2w`, `1m`, `1q`).

Two paragraphs of context for the **risk-overlay framing**:

This model is a *supplement* to AR baselines, not a replacement. It loses on
weekly RMSE versus a simple AR(4) — credit spreads are a near-random-walk
and beating that on RMSE is a fool's errand. What this model *does* do is
provide **directional signal** (~50% hit rate vs ~4% for AR baselines at
1-week horizon) and a calibrated **Trends Risk Index** that flags credit-
stress periods in advance. Use the median as a directional cue and the
band width as a regime-uncertainty gauge — not as a "predicted price."

See `USAGE.md` for interpretation guidance, `BOOTSTRAP.md` for setup.
