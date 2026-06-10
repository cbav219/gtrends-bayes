"""Step 2: remove long-term downward drift in log-SVIs via PCA on HP-trends.

The denominator of every SVI (total Google searches) has grown massively since
2004, so most categories drift downward over time even when "real" interest is
flat. The OECD Annex A fix:

1. HP-filter each log-SVI column → matrix of slow trends.
2. Run PCA on the trend matrix; keep the first principal component as the
   common drift signal.
3. Rescale PC1 so its mean and std match the cross-query mean log-SVI's
   first two moments (puts PC1 on the same scale as the data being de-drifted).
4. Subtract the rescaled PC1 from every original log-SVI column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.filters.hp_filter import hpfilter

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

_MIN_OBS_FOR_HP = 10
_MIN_DATES_FOR_PCA = 20


def _hp_trend_matrix(df_log_svi: pd.DataFrame, hp_lambda: float) -> pd.DataFrame:
    """Apply ``hpfilter(.., lamb=hp_lambda)`` per column, return trend components.

    Columns with fewer than :data:`_MIN_OBS_FOR_HP` non-NaN observations are
    dropped from the returned matrix (and logged).
    """
    trends: dict[str, pd.Series] = {}
    for col in df_log_svi.columns:
        series = df_log_svi[col].dropna()
        if len(series) < _MIN_OBS_FOR_HP:
            log.warning("skipping HP filter for %s: only %d obs", col, len(series))
            continue
        _cycle, trend = hpfilter(series, lamb=hp_lambda)
        trends[col] = trend.reindex(df_log_svi.index)
    return pd.DataFrame(trends, index=df_log_svi.index)


def _common_drift(
    df_log_svi: pd.DataFrame,
    hp_lambda: float,
) -> pd.Series:
    """Compute the rescaled first principal component of HP trends.

    Returns a Series indexed by ``df_log_svi.index``. Dates outside the all-
    columns-non-NaN window are forward/backward-filled from the nearest valid
    PC1 value so the subtraction in :func:`remove_long_term_drift` is defined
    everywhere.
    """
    trends = _hp_trend_matrix(df_log_svi, hp_lambda=hp_lambda)
    valid = trends.dropna(axis=0, how="any")
    if len(valid) < _MIN_DATES_FOR_PCA or valid.shape[1] < 1:
        log.warning(
            "not enough complete-row dates for PCA (got %d, need >= %d) — "
            "returning zero drift component",
            len(valid), _MIN_DATES_FOR_PCA,
        )
        return pd.Series(0.0, index=df_log_svi.index, name="common_drift")

    pca = PCA(n_components=1)
    pc1_values = pca.fit_transform(valid.values).ravel()
    pc1 = pd.Series(pc1_values, index=valid.index)

    # Sign convention: PC1's sign is arbitrary. Force PC1 to be positively
    # correlated with the cross-query mean trend so subtracting it removes
    # (rather than amplifies) the common drift.
    mean_trend = valid.mean(axis=1)
    if pc1.corr(mean_trend) < 0:
        pc1 = -pc1

    # Rescale to match the cross-query mean log-SVI's mean+std (per Annex A).
    mean_log_svi = df_log_svi.mean(axis=1).reindex(valid.index).dropna()
    target_mean = float(mean_log_svi.mean())
    target_std = float(mean_log_svi.std())
    if not np.isfinite(target_std) or target_std == 0:
        # Pathological case: no spread in mean log-SVI. Fall back to PC1 itself.
        rescaled = pc1 - pc1.mean() + target_mean
    else:
        pc1_z = (pc1 - pc1.mean()) / pc1.std()
        rescaled = pc1_z * target_std + target_mean
    rescaled.name = "common_drift"

    # Extend to the full index by ffill/bfill — preserves the drift level
    # outside the valid window.
    return rescaled.reindex(df_log_svi.index).ffill().bfill()


def remove_long_term_drift(
    df_log_svi: pd.DataFrame,
    hp_lambda: float = 1600.0,
) -> pd.DataFrame:
    """De-drift log-SVIs by subtracting the rescaled common HP-PCA component.

    Parameters
    ----------
    df_log_svi : pandas.DataFrame
        Wide format ``log(SVI)`` matrix (index date, columns query).
    hp_lambda : float, default 1600
        Hodrick-Prescott smoothing parameter. 1600 is the canonical quarterly
        default; for weekly Trends consider 129600. Exposed for tuning.

    Returns
    -------
    pandas.DataFrame
        De-drifted log-SVI matrix with the same shape as the input.
    """
    if df_log_svi.empty:
        return df_log_svi.copy()
    drift = _common_drift(df_log_svi, hp_lambda=hp_lambda)
    return df_log_svi.sub(drift, axis=0)


def extract_common_component(
    df_log_svi: pd.DataFrame,
    hp_lambda: float = 1600.0,
) -> pd.Series:
    """Return the rescaled PC1 of HP-filtered log-SVI trends (for diagnostics)."""
    if df_log_svi.empty:
        return pd.Series(dtype=float, name="common_drift")
    return _common_drift(df_log_svi, hp_lambda=hp_lambda)
