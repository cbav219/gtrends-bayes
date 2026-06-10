"""Step 1: average ``n_samples`` independent draws of each Trends series.

Trends responses are sampled — repeated calls for the same query return slightly
different SVI values. The OECD methodology (Annex A) recommends pulling each
query 6+ times and averaging. Series whose cross-sample standard deviation
exceeds a configurable threshold are dropped as too noisy to use.
"""

from __future__ import annotations

import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


def average_samples(
    df: pd.DataFrame,
    drop_high_variance: bool = True,
    var_threshold: float = 25.0,
) -> pd.DataFrame:
    """Average multi-sample Trends pulls into a single per-query wide series.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format ``date | query | sample_idx | svi`` (concatenation of
        ``data.trends_client.pull_series`` outputs across queries).
    drop_high_variance : bool, default True
        If True, drop any query whose cross-sample standard deviation exceeds
        ``var_threshold`` (the "too noisy to use" filter from Annex A).
    var_threshold : float, default 25
        Cross-sample standard-deviation threshold on the raw 0-100 SVI scale.
        Computed per-(date, query) cell, then averaged across dates.

    Returns
    -------
    pandas.DataFrame
        Wide format: index = date (DatetimeIndex), columns = query names,
        values = mean SVI across samples.
    """
    required = {"date", "query", "sample_idx", "svi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if drop_high_variance:
        # Cross-sample std at each (date, query), then mean across dates.
        per_cell_std = (
            df.groupby(["query", "date"])["svi"].std(ddof=0)
            .groupby(level="query").mean()
        )
        too_noisy = per_cell_std[per_cell_std > var_threshold].index.tolist()
        if too_noisy:
            log.warning(
                "dropping %d high-variance series (threshold=%.1f): %s",
                len(too_noisy), var_threshold, too_noisy,
            )
            df = df[~df["query"].isin(too_noisy)]

    means = df.groupby(["date", "query"])["svi"].mean().unstack("query")
    means.index.name = "date"
    return means.sort_index()
