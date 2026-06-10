"""Split-conformal recalibration for posterior credibility bands.

v1's universal undercoverage (~60% empirical at 80% nominal across every
model on both targets — not BSTS-specific, a global hygiene gap) closes via
a simple multiplier ``α`` applied to the half-bands after the model is fit.
No re-fit required.

The split is conformal-style: learn ``α`` on a validation slice where we can
observe coverage gaps, then apply at inference. This module exposes:

- ``learn_conformal_multiplier(y, q_low, q_high, nominal_level)`` — the core
  fitting primitive on aligned series.
- ``apply_conformal_multiplier(median, q_low, q_high, alpha)`` — inflate the
  symmetric half-bands by ``α``.
- ``fit_per_level(y, bands, levels, val_split)`` — convenience: learn ``α``
  at each of several nominal levels from a chronological train/val split.

Sign convention: ``α = 1`` is the identity. Under-coverage (empirical <
nominal) yields ``α > 1`` (inflate). Over-coverage yields ``α < 1`` (shrink).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


def learn_conformal_multiplier(
    y_actual: pd.Series,
    q_low: pd.Series,
    q_high: pd.Series,
    nominal_level: float,
    median: pd.Series | None = None,
) -> float:
    """Find the smallest ``α ≥ 0`` such that the inflated band covers ``nominal_level``.

    The inflated band is
        ``[median − α·(median − q_low), median + α·(q_high − median)]``.
    If ``median`` is omitted, it's approximated as ``(q_low + q_high) / 2``.

    Returns ``α``. Empirical coverage on the *fitting* data is exactly
    ``nominal_level`` by construction (to the granularity of the sample size).

    Parameters
    ----------
    y_actual : pandas.Series
        Realized values, date-indexed.
    q_low, q_high : pandas.Series
        Lower / upper quantile bands at some nominal level (e.g. q100, q900
        for the 80% band).
    nominal_level : float in (0, 1)
        Target coverage probability (e.g. 0.80).
    median : pandas.Series, optional
        Posterior median per date. If absent, ``(q_low + q_high) / 2`` is used.

    Returns
    -------
    float
        Multiplier ``α``. ``1.0`` means the input band already meets nominal.
    """
    common = y_actual.index.intersection(q_low.index).intersection(q_high.index)
    if median is not None:
        common = common.intersection(median.index)
    if len(common) == 0:
        return float("nan")

    y = y_actual.loc[common].astype(float)
    lo = q_low.loc[common].astype(float)
    hi = q_high.loc[common].astype(float)
    med = (lo + hi) / 2.0 if median is None else median.loc[common].astype(float)

    # Per-observation "scale" needed for that point to be inside the band:
    #   α_i = max((med_i - y_i) / (med_i - lo_i),  (y_i - med_i) / (hi_i - med_i), 0)
    half_low = (med - lo).clip(lower=1e-12)
    half_high = (hi - med).clip(lower=1e-12)
    alpha_low = ((med - y) / half_low).clip(lower=0.0)
    alpha_high = ((y - med) / half_high).clip(lower=0.0)
    alpha_per_obs = np.maximum(alpha_low, alpha_high)

    # The smallest α covering ``nominal_level`` of observations is the
    # ``nominal_level``-quantile of the per-observation needs.
    alpha = float(np.quantile(alpha_per_obs.values, nominal_level))
    return alpha


def apply_conformal_multiplier(
    median: pd.Series,
    q_low: pd.Series,
    q_high: pd.Series,
    alpha: float,
) -> tuple[pd.Series, pd.Series]:
    """Inflate the symmetric half-bands by ``α``; return (q_low_cal, q_high_cal)."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative finite float; got {alpha}")
    common = median.index.intersection(q_low.index).intersection(q_high.index)
    med = median.loc[common].astype(float)
    lo = q_low.loc[common].astype(float)
    hi = q_high.loc[common].astype(float)
    half_low = med - lo
    half_high = hi - med
    return med - alpha * half_low, med + alpha * half_high


# Map each nominal level → the (q_low_col, q_high_col) pair in the standard
# WalkForward output schema.
_LEVEL_TO_COLS: dict[float, tuple[str, str]] = {
    0.50: ("q250", "q750"),
    0.80: ("q100", "q900"),
    0.95: ("q025", "q975"),
}


def fit_per_level(
    y_actual: pd.Series,
    bands: pd.DataFrame,
    levels: Sequence[float] = (0.50, 0.80, 0.95),
    val_split: float | None = None,
    median_col: str = "q500",
) -> dict[float, dict]:
    """Learn ``α`` per nominal level.

    Two modes:

    * **In-sample** (``val_split is None``, default): use the full slice to
      learn ``α``. Reported empirical coverage is exactly ``nominal_level`` by
      construction, modulo the sample granularity. This is the right mode for
      *publishing* an α to apply at inference time — it answers "by how much
      do we need to inflate the bands to match nominal coverage on the
      forecast distribution we care about?"

    * **Out-of-sample** (``val_split in (0, 1)``): learn ``α`` on the first
      ``val_split`` fraction (chronological prefix), then apply to the
      remainder. Reports the post-recalibration coverage on the held-out test
      slice — an honesty check for whether the α generalizes across the
      regime split. Useful for the deck's "out-of-sample" coverage table.

    Both modes also report the pre-recalibration coverage on the full / test
    slice for comparison.

    Parameters
    ----------
    y_actual : pandas.Series
    bands : pandas.DataFrame
        Date-indexed; columns include the standard ``WalkForward.run`` schema
        ``q025, q050, q100, q250, q500, q750, q900, q975``.
    levels : sequence of float
    val_split : float or None
        ``None`` → in-sample calibration on the full slice (the default).
        Numeric in (0, 1) → chronological split with that fraction as val.
    median_col : str

    Returns
    -------
    dict
        ``{level: {"alpha": float, "empirical_pre_full": float,
                    "empirical_post_full": float,
                    "empirical_pre_test": float | None,
                    "empirical_post_test": float | None,
                    "n_full": int, "n_val": int | None, "n_test": int | None}}``.
    """
    common = y_actual.index.intersection(bands.index)
    if len(common) < 10:
        raise ValueError(
            f"need at least 10 overlapping observations; got {len(common)}"
        )
    y = y_actual.loc[common].sort_index()
    b = bands.loc[y.index].sort_index()
    median_full = b[median_col]

    val_indices = None
    test_indices = None
    if val_split is not None:
        cut = int(np.floor(len(y) * val_split))
        if cut < 5 or len(y) - cut < 5:
            raise ValueError(
                f"val_split={val_split} produces too-small slices "
                f"(val={cut}, test={len(y) - cut})"
            )
        val_indices = y.index[:cut]
        test_indices = y.index[cut:]

    out: dict[float, dict] = {}
    for level in levels:
        if level not in _LEVEL_TO_COLS:
            raise ValueError(f"no column mapping for level={level}")
        lo_col, hi_col = _LEVEL_TO_COLS[level]
        if lo_col not in b.columns or hi_col not in b.columns:
            log.warning("bands missing %s/%s; skipping level=%.2f", lo_col, hi_col, level)
            continue

        lo_full = b[lo_col]
        hi_full = b[hi_col]
        pre_full = float(((y >= lo_full) & (y <= hi_full)).mean())

        # In-sample α: learned on the full slice (the published value).
        alpha = learn_conformal_multiplier(
            y, lo_full, hi_full, nominal_level=level, median=median_full,
        )
        lo_cal, hi_cal = apply_conformal_multiplier(median_full, lo_full, hi_full, alpha)
        post_full = float(((y >= lo_cal) & (y <= hi_cal)).mean())

        row = {
            "alpha": float(alpha),
            "empirical_pre_full": pre_full,
            "empirical_post_full": post_full,
            "n_full": int(len(y)),
        }

        # Optional OOS robustness check.
        if val_split is not None:
            assert val_indices is not None and test_indices is not None
            y_val = y.loc[val_indices]
            y_test = y.loc[test_indices]
            median_val = median_full.loc[val_indices]
            median_test = median_full.loc[test_indices]
            lo_val, hi_val = lo_full.loc[val_indices], hi_full.loc[val_indices]
            lo_test, hi_test = lo_full.loc[test_indices], hi_full.loc[test_indices]

            pre_test = float(((y_test >= lo_test) & (y_test <= hi_test)).mean())
            alpha_oos = learn_conformal_multiplier(
                y_val, lo_val, hi_val, nominal_level=level, median=median_val,
            )
            lo_test_cal, hi_test_cal = apply_conformal_multiplier(
                median_test, lo_test, hi_test, alpha_oos,
            )
            post_test = float(((y_test >= lo_test_cal) & (y_test <= hi_test_cal)).mean())

            row.update({
                "alpha_oos": float(alpha_oos),
                "empirical_pre_test": pre_test,
                "empirical_post_test": post_test,
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
            })
        else:
            row.update({
                "alpha_oos": None,
                "empirical_pre_test": None,
                "empirical_post_test": None,
                "n_val": None,
                "n_test": None,
            })

        out[level] = row
    return out
