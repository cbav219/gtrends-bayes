"""Pure-Python inference layer for the frozen gtrends-bayes model bundle.

What this module does
---------------------
Loads a frozen BSTS (Bayesian Structural Time Series) + AR(p) model and
produces a horizon-step-ahead point forecast plus 90% credible band for
HY / IG corporate-bond ETF prices (HYG / LQD — the proxies the project
uses for HY / IG OAS spreads).

The model combines:

* an AR(p) backbone trained on the weekly target series, and
* a BSTS regression on a 41-column Google Trends predictor matrix
  (preprocessed per the OECD Annex A methodology).

Posterior bands are recalibrated with a per-target conformal multiplier α
(learned on a validation slice) so the 90% band achieves nominal coverage.

What this module does NOT do
----------------------------
* Re-fit the BSTS posterior. The pickle is frozen — sampling is done by
  drawing β ~ Bernoulli(inclusion_prob) · 𝒩(mean, sd) per predictor, a
  Gaussian-mixture approximation of the spike-and-slab posterior.
* Pull data. The caller is responsible for the data sideband (HY/IG
  history CSV + preprocessed Trends parquet).
* Require R, rpy2, or MCMC at runtime. Pure NumPy / pandas.

Public API
----------
::

    from gtrends_bayes.inference import load_model, forecast
    model = load_model("model/HY_v5.pkl")
    out = forecast(model, horizon="1m", as_of=pd.Timestamp("2026-05-15"),
                   y_history=y, x_latest=x)
    print(out["level_median"], out["level_band"])

See :mod:`gtrends_bayes.inference.forecast` for the full parameter list,
``docs/v5/USAGE.md`` for the PM-facing walkthrough, and
``docs/v5/BOOTSTRAP.md`` for setup on the JupyterLab VM.
"""

from .forecast import forecast
from .load import load_model

__all__ = ["load_model", "forecast"]
