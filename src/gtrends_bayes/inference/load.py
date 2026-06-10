"""Load + validate a frozen gtrends-bayes model pickle.

The frozen pickle is produced by ``scripts/freeze_model_v4.py``
(invoked with ``--bundle-version v5`` for the v5 ship) from a trained
BSTS posterior.

Pickle schema (verified by :func:`load_model` on every call)
------------------------------------------------------------
Top-level keys (all required):

* ``"target"`` — ``"HY"`` or ``"IG"`` (matches ``config/targets.yaml``).
* ``"target_transform"`` — ``"levels"``, ``"diff"``, or ``"log_diff"``;
  determines the inverse-transform path inside ``forecast()``.
* ``"ar_backbone"`` — dict with ``{p, coefficients, intercept, sigma}``;
  AR(p) parameters fitted on the transformed target.
* ``"bsts_posterior"`` — dict with at least
  ``{inclusion_probs, coefficient_summary, X_columns}``; spike-and-slab
  summary statistics. The full MCMC draws are *not* stored (kept the
  pickle small).
* ``"preprocessing"`` — dict with ``cadence``, ``yoy_periods_per_year``,
  ``structural_break_dates`` and any learned PCA/HP-filter state.

Optional but typically present:

* ``"conformal_alpha"`` — float; multiplier applied to the 90% band so it
  achieves nominal coverage on the validation slice. Defaults to 1.0 if
  absent.
* ``"build_timestamp"``, ``"v3_commit_hash"`` — traceability metadata.
* ``"history_file"`` — filename of the matching y-history CSV inside the
  data sideband (e.g. ``"HY_history.csv"``). Lets verify_data.py and
  example_forecast.py pair each pickle with the right data file
  automatically.
* ``"oas_overlay_translation"`` — dict (ETF targets only). Empirical ETF↔OAS
  regression baked in at freeze time so :func:`forecast` can emit
  ``oas_implied_median`` / ``oas_implied_band`` / ``oas_implied_path_*``
  alongside the level-space ETF forecast. Schema::

      {
        "slope_bps_per_dlog": float,   # OLS slope of ΔOAS-bps on ETF-Δlog
        "pearson": float,              # overlap-window Pearson correlation
        "spearman": float,
        "n_overlap_weeks": int,
        "overlap_start": str,          # ISO date
        "overlap_end": str,
        "last_oas_bps": float,         # latest weekly OAS at freeze time
        "last_oas_date": str,          # ISO date
        "proxy_quality_label": str,    # "defensible" | "moderate" | "weak"
        "source": str,
      }

  Absent for OAS-direct targets (``HY_OAS`` / ``IG_OAS``) — those forecast
  bps natively and need no translation.

Any pickle missing the required keys raises ``ValueError`` with a clear
message pointing at what's wrong.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Top-level keys the freeze script writes. Any pickle missing one of
# these is malformed and won't satisfy forecast().
_REQUIRED_KEYS: frozenset[str] = frozenset({
    "target",
    "target_transform",
    "ar_backbone",
    "bsts_posterior",
    "preprocessing",
})

# Sub-keys we explicitly need from ar_backbone and bsts_posterior. The
# rest of the pickle's content is allowed to vary across model versions.
_REQUIRED_AR_KEYS: frozenset[str] = frozenset({"p", "coefficients", "intercept", "sigma"})
_REQUIRED_BSTS_KEYS: frozenset[str] = frozenset({
    "inclusion_probs", "coefficient_summary", "X_columns",
})


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a frozen model pickle and validate its schema.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a pickle written by ``scripts/freeze_model_v4.py``
        (invoked with ``--bundle-version v5`` for the v5 ship).
        Typically ``model/HY_v5.pkl`` or ``model/IG_v5.pkl`` inside the
        unpacked bundle.

    Returns
    -------
    dict
        The unpickled model. See module docstring for the schema.

    Raises
    ------
    FileNotFoundError
        Pickle file doesn't exist at ``path``.
    ValueError
        Pickle doesn't unpickle to a dict, is missing required top-level
        or sub-keys, or has an unrecognized ``target_transform`` value.

    Examples
    --------
    >>> model = load_model("model/HY_v5.pkl")
    >>> model["target"]
    'HY'
    >>> model["ar_backbone"]["p"]
    4
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"frozen model file not found: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, dict):
        raise ValueError(
            f"frozen model at {path} should unpickle to dict; got {type(model).__name__}"
        )

    # Defense-in-depth: validate the schema before any downstream code
    # tries to dereference these keys. Clear error > obscure KeyError.
    missing = _REQUIRED_KEYS - set(model.keys())
    if missing:
        raise ValueError(
            f"frozen model {path} missing required top-level keys: {sorted(missing)}"
        )

    ar_missing = _REQUIRED_AR_KEYS - set(model["ar_backbone"].keys())
    if ar_missing:
        raise ValueError(
            f"frozen model {path} ar_backbone missing keys: {sorted(ar_missing)}"
        )

    bsts_missing = _REQUIRED_BSTS_KEYS - set(model["bsts_posterior"].keys())
    if bsts_missing:
        raise ValueError(
            f"frozen model {path} bsts_posterior missing keys: {sorted(bsts_missing)}"
        )

    if model["target_transform"] not in ("levels", "diff", "log_diff"):
        raise ValueError(
            f"frozen model target_transform={model['target_transform']!r} "
            "not in ('levels', 'diff', 'log_diff')"
        )

    return model
