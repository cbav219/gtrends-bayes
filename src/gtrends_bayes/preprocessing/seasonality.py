"""Step 3: kill seasonality with year-over-year log differences.

Categories: YoY log-difference (seasonal patterns are strong).
Topics: kept in log-levels (less seasonal, more event-driven).

Inputs are assumed to already be in log-SVI space (bias_removal happens before
this step in the pipeline).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

QueryClass = Literal["category", "topic"]

log = get_logger(__name__)

# Triangular weights for the OECD weekly-tracker variant: weighted average of
# the prior-year values at offsets t-53, t-52, t-51 (a smoothing trick from
# Annex A bottom — handles the small calendar jitter between this year and
# last year's "same week").
_NEIGHBOR_OFFSETS: tuple[int, int, int] = (-1, 0, 1)
_NEIGHBOR_WEIGHTS: tuple[float, float, float] = (0.25, 0.5, 0.25)


def yoy_log_diff(
    df: pd.DataFrame,
    periods_per_year: int = 52,
    weighted_neighbor: bool = True,
) -> pd.DataFrame:
    """Compute year-over-year log differences on every column.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide format, date-indexed, values in log-SVI space.
    periods_per_year : int, default 52
        Lag (in periods) to use as "same week, prior year".
    weighted_neighbor : bool, default True
        If True, the prior-year reference is a weighted average of the values
        at lags ``periods_per_year`` ± 1, with weights (0.25, 0.5, 0.25). If
        False, the simple ``x_t - x_{t-periods_per_year}`` form is used.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``df``; the first ``periods_per_year + 1`` rows (52 for
        weekly data) are NaN because the YoY reference is unavailable there.
    """
    if df.empty:
        return df.copy()

    if not weighted_neighbor:
        return df - df.shift(periods_per_year)

    # Build a weighted prior-year reference for every column.
    ref = sum(
        w * df.shift(periods_per_year - off)
        for off, w in zip(_NEIGHBOR_OFFSETS, _NEIGHBOR_WEIGHTS)
    )
    return df - ref


def transform_by_class(
    df: pd.DataFrame,
    classes: dict[str, QueryClass],
    periods_per_year: int = 52,
    weighted_neighbor: bool = True,
) -> pd.DataFrame:
    """Apply YoY-log-diff to category columns; pass topic columns through.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide format, date-indexed, log-SVI.
    classes : dict[str, {"category", "topic"}]
        Maps each column name in ``df`` to its query class. Columns not in
        ``classes`` are treated as ``"category"`` (the conservative default —
        differencing a topic is harmless on stationary-ish series).
    periods_per_year : int, default 52
    weighted_neighbor : bool, default True

    Returns
    -------
    pandas.DataFrame
        Same shape as ``df``. Category columns are YoY-differenced; topic
        columns are returned in log-levels.
    """
    if df.empty:
        return df.copy()

    cat_cols = [c for c in df.columns if classes.get(c, "category") == "category"]
    topic_cols = [c for c in df.columns if classes.get(c, "category") == "topic"]

    parts: list[pd.DataFrame] = []
    if cat_cols:
        parts.append(
            yoy_log_diff(df[cat_cols], periods_per_year=periods_per_year,
                         weighted_neighbor=weighted_neighbor)
        )
    if topic_cols:
        parts.append(df[topic_cols].copy())
    if not parts:
        return df.copy()

    out = pd.concat(parts, axis=1)
    # Restore original column order.
    return out[df.columns]
