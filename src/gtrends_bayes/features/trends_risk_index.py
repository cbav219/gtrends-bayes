"""Trends Risk Index — the publishable v2 deliverable.

Combines a fitted BSTS posterior (HY or IG) with the historical preprocessed
Trends matrix into a single weighted "stress signal" series. Output is
z-scored over a rolling 5-year window so the value at any week reads as
"how stress-leaning is this week relative to the last five years".

Sign convention: **positive = stress-leaning** (predicted spread widening).
For ETF *price* targets (HYG, LQD) we flip the sign internally because
price-down ≡ spread-up; pass ``target_kind='spread'`` if your target is an
OAS / yield-spread series.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

_ROLLING_WINDOW_WEEKS = 5 * 52   # 5 years of weekly bars
_MIN_PERIODS_FOR_ZSCORE = 52      # need at least 1y of history before z-scoring
_ROLLING_WINDOW_DAYS = 5 * 252    # 5 years of business days
_MIN_PERIODS_FOR_ZSCORE_DAILY = 252


def build_risk_index(
    posterior: dict,
    X_history: pd.DataFrame,
    target_kind: Literal["price", "spread"] = "price",
    rolling_window_weeks: int = _ROLLING_WINDOW_WEEKS,
    min_periods: int = _MIN_PERIODS_FOR_ZSCORE,
    tier_thresholds: tuple[float, float] = (-1.0, 1.0),
    cadence: Literal["weekly", "daily"] = "weekly",
    rolling_window_periods: int | None = None,
) -> pd.DataFrame:
    """Build the Trends Risk Index for a single target.

    Per-predictor per-week contribution = ``P(γ_j=1) · β̄_j · X_{j,t}``;
    summed across predictors gives the raw weighted prediction. The flip for
    ``target_kind="price"`` makes positive = stress-leaning regardless of
    whether your target is an OAS spread or an ETF price.

    Parameters
    ----------
    posterior : dict
        Loaded from ``data/processed/posterior/{HY,IG}_bsts_v1.pkl``. Must
        contain ``"coefficient_summary"`` (DataFrame indexed by predictor
        name with at least ``inclusion_prob`` and ``mean_when_included``).
    X_history : pandas.DataFrame
        Processed Trends features (output of ``Pipeline.fit_transform``,
        post-quality-filter). Date-indexed; columns are predictor names.
    target_kind : {"price", "spread"}, default "price"
        Determines sign convention. Pass ``"spread"`` if forecasting an OAS /
        yield-spread series; ``"price"`` is correct for HYG/LQD ETF targets.
    rolling_window_weeks : int, default 260
        Window for the rolling z-score (5y).
    min_periods : int, default 52
        Minimum observations before z-score is computed.
    tier_thresholds : (low, high), default (-1.0, 1.0)
        z-score cut-offs for the ``tier`` label ("low" / "med" / "high").

    Returns
    -------
    pandas.DataFrame
        Date-indexed with columns ``raw_index``, ``zscore_5y``, ``tier``,
        ``contribution_top_predictor``, ``contribution_top_value``.
    """
    if target_kind not in ("price", "spread"):
        raise ValueError(f"unknown target_kind: {target_kind!r}")
    if cadence not in ("weekly", "daily"):
        raise ValueError(f"unknown cadence: {cadence!r}")

    # Cadence-aware rolling-z-score window. The weekly-only API is preserved
    # for back-compat: callers passing rolling_window_weeks but no
    # rolling_window_periods get the legacy behavior.
    if rolling_window_periods is not None:
        eff_window = int(rolling_window_periods)
        eff_min = int(min_periods)
    elif cadence == "daily":
        eff_window = _ROLLING_WINDOW_DAYS
        eff_min = _MIN_PERIODS_FOR_ZSCORE_DAILY
    else:
        eff_window = int(rolling_window_weeks)
        eff_min = int(min_periods)

    summary = posterior["coefficient_summary"]
    if "mean_when_included" not in summary.columns:
        raise ValueError("posterior['coefficient_summary'] missing 'mean_when_included'")
    if "inclusion_prob" not in summary.columns:
        raise ValueError("posterior['coefficient_summary'] missing 'inclusion_prob'")

    # Restrict to predictors that exist in BOTH the posterior and X_history.
    common_cols = [c for c in summary.index if c in X_history.columns]
    if not common_cols:
        raise ValueError("no overlap between posterior predictors and X_history columns")
    log.info("trends_risk_index: %d/%d predictors usable", len(common_cols), len(summary))

    inclusion = summary.loc[common_cols, "inclusion_prob"].astype(float)
    beta_when = summary.loc[common_cols, "mean_when_included"].astype(float).fillna(0.0)
    X = X_history[common_cols].astype(float)

    # Per-predictor per-week contribution: P(γ) · β̄ · X.
    weights = inclusion * beta_when                  # signed magnitudes
    contributions = X.mul(weights, axis=1)            # shape (T, J)

    raw = contributions.sum(axis=1)

    # ETF price target → flip so positive = stress (price-down equivalence).
    if target_kind == "price":
        raw = -raw
        contributions = -contributions
    raw.name = "raw_index"

    # Rolling 5-year z-score (cadence-aware window).
    rolling = raw.rolling(window=eff_window, min_periods=eff_min)
    zscore = (raw - rolling.mean()) / rolling.std()
    zscore.name = "zscore_5y"

    # Tier labels.
    low_thr, high_thr = tier_thresholds
    tier = pd.Series("med", index=raw.index, name="tier")
    tier[zscore <= low_thr] = "low"
    tier[zscore >= high_thr] = "high"

    # Top-contributing predictor per week (vectorized lookup).
    abs_contrib = contributions.abs()
    top_pred = abs_contrib.idxmax(axis=1)
    contrib_arr = contributions.values
    col_codes = pd.Categorical(top_pred, categories=contributions.columns).codes
    valid = col_codes >= 0
    top_val_arr = np.full(len(contributions), np.nan)
    if valid.any():
        rows = np.arange(len(contributions))[valid]
        top_val_arr[valid] = contrib_arr[rows, col_codes[valid]]
    top_val = pd.Series(top_val_arr, index=contributions.index)

    return pd.DataFrame({
        "raw_index": raw,
        "zscore_5y": zscore,
        "tier": tier,
        "contribution_top_predictor": top_pred,
        "contribution_top_value": top_val,
    })


def crisis_windows() -> dict[str, pd.Timestamp]:
    """Anchor dates for the recall-against-crises evaluation."""
    return {
        "COVID 2020-03": pd.Timestamp("2020-03-15"),
        "UK gilt 2022-09": pd.Timestamp("2022-09-25"),
        "SVB 2023-03": pd.Timestamp("2023-03-12"),
    }
