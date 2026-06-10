"""Backtest scoring metrics.

The v1 metrics (rmse, mae, standardized_rmse, rmse_ratio, directional_hit_rate,
posterior_coverage) remain for completeness. The v2 reframe (see
``IMPLEMENTATION_PLAN_v2.md`` §2) shifts the headline evaluation to **information
coefficient + precision/recall on widening events** because BSTS at one-week
horizon does not beat AR(4) on RMSE while AR / Naïve / AR+VIX cluster within
±1% of each other on RMSE — there is no point-forecast frontier to beat at
this horizon. Directional accuracy and event-prediction precision/recall are
where Trends-driven models actually win.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error over the common (non-NaN) index."""
    common = y_true.index.intersection(y_pred.index)
    err = (y_true.loc[common].astype(float) - y_pred.loc[common].astype(float)).dropna()
    if err.empty:
        return float("nan")
    return float(np.sqrt((err**2).mean()))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error over the common (non-NaN) index."""
    common = y_true.index.intersection(y_pred.index)
    err = (y_true.loc[common].astype(float) - y_pred.loc[common].astype(float)).dropna()
    if err.empty:
        return float("nan")
    return float(err.abs().mean())


def standardized_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """``rmse(y_true, y_pred) / std(y_true)`` over the same index."""
    common = y_true.index.intersection(y_pred.index)
    y = y_true.loc[common].astype(float).dropna()
    if y.empty or y.std(ddof=0) == 0:
        return float("nan")
    return rmse(y_true, y_pred) / float(y.std(ddof=0))


def rmse_ratio(y_true: pd.Series, y_pred: pd.Series, y_pred_baseline: pd.Series) -> float:
    """``rmse(y_true, y_pred) / rmse(y_true, y_pred_baseline)``. <1 = beats baseline."""
    base = rmse(y_true, y_pred_baseline)
    if base == 0 or np.isnan(base):
        return float("nan")
    return rmse(y_true, y_pred) / base


def directional_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of forecast steps where ``sign(Δy_pred) == sign(Δy_true)``.

    "Δ" is the change vs. the previous y value. NaN-safe.
    """
    common = y_true.index.intersection(y_pred.index)
    y_t = y_true.loc[common].astype(float)
    y_p = y_pred.loc[common].astype(float)
    dt = y_t.diff().dropna()
    dp = y_p.diff().dropna()
    common2 = dt.index.intersection(dp.index)
    if len(common2) == 0:
        return float("nan")
    matches = (np.sign(dt.loc[common2]) == np.sign(dp.loc[common2]))
    return float(matches.mean())


def posterior_coverage(
    y_true: pd.Series,
    quantile_bands: pd.DataFrame,
    levels: Sequence[float] = (0.50, 0.80, 0.95),
) -> dict[float, float]:
    """Empirical coverage of the posterior bands at each nominal level.

    Parameters
    ----------
    y_true : pandas.Series
        Realized values, indexed by forecast date.
    quantile_bands : pandas.DataFrame
        Indexed by forecast date with columns named ``q025``, ``q975``, etc.
        (matching what ``WalkForward.run`` writes).
    levels : iterable of float
        Nominal credibility levels to score.

    Returns
    -------
    dict
        ``{level: empirical_coverage}``. NaN if the requested band columns
        are absent.
    """
    out: dict[float, float] = {}
    common = y_true.index.intersection(quantile_bands.index)
    if len(common) == 0:
        return {lvl: float("nan") for lvl in levels}
    y = y_true.loc[common].astype(float)
    bands = quantile_bands.loc[common]
    for lvl in levels:
        q_low_name = f"q{int(round((0.5 - lvl / 2) * 1000)):03d}"
        q_high_name = f"q{int(round((0.5 + lvl / 2) * 1000)):03d}"
        if q_low_name not in bands.columns or q_high_name not in bands.columns:
            out[lvl] = float("nan")
            continue
        inside = (y >= bands[q_low_name]) & (y <= bands[q_high_name])
        out[lvl] = float(inside.mean())
    return out


# ============================================================================
# v2 metrics (Phase A — IMPLEMENTATION_PLAN_v2.md §2)
# ============================================================================


def brier_score(p_pred: pd.Series, y_event: pd.Series) -> float:
    """Mean squared error between predicted probability and binary outcome.

    Lower is better. ``0.0`` is perfect, ``0.25`` is the uninformative reference
    (predicting 0.5 always), ``1.0`` is anti-correct (probability assigned to
    the wrong class with full confidence).

    Parameters
    ----------
    p_pred : pandas.Series of float in [0, 1]
        Predicted probability that the positive event occurs.
    y_event : pandas.Series of {0, 1}
        Realized binary outcome.
    """
    common = p_pred.index.intersection(y_event.index)
    p = p_pred.loc[common].astype(float).dropna()
    y = y_event.loc[common].astype(float).dropna()
    common = p.index.intersection(y.index)
    if len(common) == 0:
        return float("nan")
    p, y = p.loc[common], y.loc[common]
    return float(((p - y) ** 2).mean())


def auc_roc(p_pred: pd.Series, y_event: pd.Series) -> float:
    """Area under the ROC curve for a binary direction classifier.

    Uses ``sklearn.metrics.roc_auc_score``. Returns NaN when ``y_event`` has
    only one class on the common index (AUC is undefined there).
    """
    from sklearn.metrics import roc_auc_score

    common = p_pred.index.intersection(y_event.index)
    p = p_pred.loc[common].astype(float).dropna()
    y = y_event.loc[common].astype(float).dropna()
    common = p.index.intersection(y.index)
    if len(common) < 2:
        return float("nan")
    p, y = p.loc[common], y.loc[common].astype(int)
    if y.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y.values, p.values))
    except ValueError:
        return float("nan")


def information_coefficient(
    y_pred: pd.Series,
    y_actual: pd.Series,
    method: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """Rank correlation between predicted and realized changes.

    The standard quant-research IC. Range ``[-1, 1]``; >0.05 is meaningful at
    weekly horizons in credit. Computed on first-differences (``Δy``) of both
    series, intersected on common dates, NaN-safe.

    Parameters
    ----------
    y_pred, y_actual : pandas.Series
        Date-indexed level series; first-differenced internally.
    method : {"spearman", "pearson"}, default "spearman"
        Spearman is the convention for IC because credit returns are heavy-tailed.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unknown method: {method!r}")

    from scipy import stats

    common = y_pred.index.intersection(y_actual.index)
    if len(common) < 3:
        return float("nan")
    dp = y_pred.loc[common].astype(float).diff().dropna()
    da = y_actual.loc[common].astype(float).diff().dropna()
    common2 = dp.index.intersection(da.index)
    if len(common2) < 3:
        return float("nan")
    dp_v, da_v = dp.loc[common2].values, da.loc[common2].values
    if np.std(dp_v) == 0 or np.std(da_v) == 0:
        return float("nan")
    if method == "spearman":
        rho, _ = stats.spearmanr(dp_v, da_v)
    else:  # pearson
        rho, _ = stats.pearsonr(dp_v, da_v)
    return float(rho) if np.isfinite(rho) else float("nan")


def conditional_hit_rate(
    y_pred: pd.Series,
    y_actual: pd.Series,
    move_threshold: float,
) -> dict:
    """Hit rate restricted to weeks where ``|Δy_actual| > move_threshold``.

    A directional model that's right when moves are big is more useful than one
    that's right on noise weeks. Raises ``ValueError`` if no eligible weeks
    exist (the threshold is too high for the data).

    Parameters
    ----------
    y_pred, y_actual : pandas.Series
        Level series, date-indexed.
    move_threshold : float
        Threshold on ``|Δy_actual|`` (in the same units as y_actual).

    Returns
    -------
    dict
        ``{"hit_rate": float, "n_eligible": int, "n_total": int}``.
    """
    common = y_pred.index.intersection(y_actual.index)
    if len(common) < 2:
        raise ValueError("need at least 2 overlapping observations to diff")
    dp = y_pred.loc[common].astype(float).diff().dropna()
    da = y_actual.loc[common].astype(float).diff().dropna()
    common2 = dp.index.intersection(da.index)
    eligible = common2[da.loc[common2].abs() > move_threshold]
    n_total = len(common2)
    n_eligible = len(eligible)
    if n_eligible == 0:
        raise ValueError(
            f"no weeks with |Δy_actual| > {move_threshold} in the {n_total}-week sample"
        )
    matches = (np.sign(dp.loc[eligible]) == np.sign(da.loc[eligible]))
    return {
        "hit_rate": float(matches.mean()),
        "n_eligible": int(n_eligible),
        "n_total": int(n_total),
    }


def precision_recall_widening(
    y_pred_widening: pd.Series,
    y_actual: pd.Series,
    widening_threshold: float = 25.0,
    direction: Literal["increase", "decrease", "either"] = "decrease",
) -> dict:
    """Precision / recall / F1 on the binary widening-event prediction.

    Parameters
    ----------
    y_pred_widening : pandas.Series of bool or {0, 1}
        Model's prediction that a widening event occurs in the period.
    y_actual : pandas.Series of float
        Realized levels. The positive class is derived as
        ``Δy_actual`` exceeding ``widening_threshold`` in the chosen direction.
    widening_threshold : float, default 25
        Magnitude of the move that defines an "event".
    direction : {"increase", "decrease", "either"}, default "decrease"
        Convention for what counts as "widening":
            * ``"increase"`` — Δy_actual > +threshold (use for OAS / yield-spread targets).
            * ``"decrease"`` — Δy_actual < -threshold (use for ETF *price* targets,
              where price-down = spread-up).
            * ``"either"`` — |Δy_actual| > threshold (any large move).

    Returns
    -------
    dict
        ``{"precision": float, "recall": float, "f1": float,
           "n_events": int, "n_predicted": int}``.
    """
    from sklearn.metrics import precision_recall_fscore_support

    common = y_pred_widening.index.intersection(y_actual.index)
    if len(common) < 2:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_events": 0, "n_predicted": 0}
    pred = y_pred_widening.loc[common].astype(int)
    da = y_actual.loc[common].astype(float).diff()
    common2 = pred.index.intersection(da.dropna().index)
    pred = pred.loc[common2]
    da = da.loc[common2]
    if direction == "increase":
        y_event = (da > widening_threshold).astype(int)
    elif direction == "decrease":
        y_event = (da < -widening_threshold).astype(int)
    elif direction == "either":
        y_event = (da.abs() > widening_threshold).astype(int)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    n_events = int(y_event.sum())
    n_predicted = int(pred.sum())
    if n_events == 0:
        # No positives to recall. F1 / precision undefined.
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_events": 0, "n_predicted": n_predicted}
    p, r, f, _ = precision_recall_fscore_support(
        y_event.values, pred.values, average="binary", zero_division=0,
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "n_events": n_events,
        "n_predicted": n_predicted,
    }


def coverage_calibration(
    y_actual: pd.Series,
    q_low: pd.Series,
    q_high: pd.Series,
    nominal_level: float,
) -> dict:
    """Empirical coverage of a credible interval vs its nominal level.

    Parameters
    ----------
    y_actual : pandas.Series
    q_low, q_high : pandas.Series
        Lower and upper quantile bands corresponding to ``nominal_level``.
    nominal_level : float in (0, 1)
        e.g. 0.80 for the 80% credible interval.

    Returns
    -------
    dict
        ``{"empirical": float, "nominal": float, "gap": float}``. Gap is
        ``nominal - empirical`` — positive gap means under-coverage.
    """
    common = y_actual.index.intersection(q_low.index).intersection(q_high.index)
    if len(common) == 0:
        return {"empirical": float("nan"), "nominal": float(nominal_level),
                "gap": float("nan")}
    y = y_actual.loc[common].astype(float)
    lo = q_low.loc[common].astype(float)
    hi = q_high.loc[common].astype(float)
    inside = (y >= lo) & (y <= hi)
    emp = float(inside.mean())
    return {
        "empirical": emp,
        "nominal": float(nominal_level),
        "gap": float(nominal_level - emp),
    }
