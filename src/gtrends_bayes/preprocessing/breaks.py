"""Step 4: correct the January 2011 and January 2016 collection-method breaks.

Google changed the Trends data-collection process in those Januaries; series
levels jump artificially. Fix per OECD Annex A: subtract ``(value at break) −
(value 12 months prior)`` from all observations after each break date, then
exclude 2011 and 2016 from training (~12% of the sample).

Notes
-----
* The "value at break" / "value 12 months prior" lookups use ``asof`` semantics
  so they tolerate week-of-year jitter (e.g. a date stamped 2011-01-02 vs.
  2011-01-09).
* Corrections compound across breaks, applied in chronological order.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from gtrends_bayes.logging import get_logger

DEFAULT_BREAKS = ("2011-01-01", "2016-01-01")
EXCLUDED_YEARS = (2011, 2016)
_LOOKBACK_WINDOW = pd.Timedelta(days=14)  # pad ±2 weeks for asof lookups

log = get_logger(__name__)


def _value_at_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    """Return the value at the latest index ≤ ``target``, or None if absent."""
    idx = series.index[series.index <= target]
    if len(idx) == 0:
        return None
    return float(series.loc[idx[-1]])


def _value_near(series: pd.Series, target: pd.Timestamp, window: pd.Timedelta) -> float | None:
    """Return the value at the index closest to ``target`` within ±``window``."""
    candidates = series.dropna()
    if candidates.empty:
        return None
    diffs = (candidates.index - target).to_series().abs()
    if diffs.min() > window:
        return None
    return float(candidates.iloc[diffs.values.argmin()])


def correct_jan_breaks(
    df: pd.DataFrame,
    break_dates: Iterable[str | pd.Timestamp] = DEFAULT_BREAKS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply January-break corrections; return corrected df + training mask.

    Parameters
    ----------
    df : pandas.DataFrame
        Date-indexed wide-format dataframe (post-seasonality step). Values
        are typically YoY-log-differenced category series (or log-level
        topic series).
    break_dates : iterable of str or Timestamp, default ("2011-01-01", "2016-01-01")
        Dates to treat as breaks. Each break must be inside ``df.index``'s
        span; out-of-range breaks are silently skipped.

    Returns
    -------
    corrected : pandas.DataFrame
        Same shape as ``df``, with each post-break segment translated so the
        12-month change at the break equals zero (per Annex A: "translate
        post-break series so January 2011 (and 2016) growth = 0").
    train_eligible : pandas.Series of bool
        Indexed like ``df.index``. False for any date in ``EXCLUDED_YEARS``,
        True elsewhere.
    """
    if df.empty:
        return df.copy(), pd.Series(dtype=bool)

    sorted_breaks = sorted(pd.to_datetime(d) for d in break_dates)

    out = df.copy()
    for col in out.columns:
        series = out[col]
        for brk in sorted_breaks:
            if brk < series.index.min() or brk > series.index.max():
                continue
            v_at = _value_near(series, brk, _LOOKBACK_WINDOW)
            v_prior = _value_near(series, brk - pd.DateOffset(years=1), _LOOKBACK_WINDOW)
            if v_at is None or v_prior is None:
                log.debug("break correction skipped for %s at %s: missing reference", col, brk.date())
                continue
            delta = v_at - v_prior
            mask = series.index >= brk
            series.loc[mask] = series.loc[mask] - delta
        out[col] = series

    train_eligible = pd.Series(
        ~out.index.year.isin(EXCLUDED_YEARS),
        index=out.index,
        name="train_eligible",
    )
    return out, train_eligible
