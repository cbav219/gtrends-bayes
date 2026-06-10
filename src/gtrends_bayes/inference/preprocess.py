"""Inference-time validation + column-alignment for the X matrix.

Why this file is light-weight
-----------------------------
The v5 data-sideband contract (``docs/v5/data_README.md``) says that
``trends.parquet`` is **already** multi-sampled and preprocessed in the
same shape BSTS was trained on (HP-filter drift removal, YoY differencing,
structural-break drops, low-quality column drops, multi-sample averaging).
That preprocessing is heavy (HP-filter + neighbor smoothing) and would
require shipping the entire training pipeline inside the v5 bundle.

So this module's job at inference time is purely *defensive*:

1. Confirm the caller's ``x_latest`` carries every predictor the frozen
   model expects (``model.bsts_posterior.X_columns``).
2. Re-order columns to the canonical sequence so the BSTS-residual
   matrix-multiply lines up correctly.
3. Sanity-check the cadence implied by the index frequency against the
   cadence the model was trained at — warn if they disagree.

No HP filtering, no re-fitting PCA, no Trends API calls. The caller is
responsible for shipping a parquet that matches the contract.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


def apply_preprocessing(
    X_raw: pd.DataFrame,
    preprocessing_state: dict[str, Any],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Align inference-time X to the frozen-model's expected schema.

    Parameters
    ----------
    X_raw : pandas.DataFrame
        Already-preprocessed Trends data (per the data-sideband contract).
        Indexed by date, columns = canonical predictor names.
    preprocessing_state : dict
        The ``model["preprocessing"]`` sub-dict from the frozen pickle.
        Used here only for the cadence sanity check
        (``preprocessing_state["cadence"]`` ∈ ``{"weekly", "daily"}``).
    expected_columns : list of str, optional
        If given, ``X_raw`` must carry every name in this list; the
        returned DataFrame is re-ordered to match. If a column is
        missing, raises ``ValueError`` with the missing names listed.

    Returns
    -------
    pandas.DataFrame
        Same values as input, possibly with columns re-ordered to match
        ``expected_columns``. No values are recomputed.

    Raises
    ------
    TypeError
        ``X_raw`` is not a pandas DataFrame.
    ValueError
        ``X_raw`` is missing one or more expected predictor columns.

    Warns
    -----
    UserWarning
        Index frequency disagrees with ``preprocessing_state["cadence"]``
        (e.g. caller passed a weekly X to a daily-trained model).
    """
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError(f"X_raw must be DataFrame; got {type(X_raw).__name__}")

    # Column alignment. The BSTS coefficient summary is indexed by the
    # canonical X_columns order; if we forward x_latest in a different
    # order, the matrix multiplication in forecast() silently produces
    # garbage. Re-index here so callers can't accidentally shuffle.
    if expected_columns is not None:
        missing = set(expected_columns) - set(X_raw.columns)
        if missing:
            raise ValueError(
                f"X_raw missing expected predictor columns: {sorted(missing)}. "
                f"Got {len(X_raw.columns)} columns; expected {len(expected_columns)}. "
                "Are you using the matching data sideband for this model bundle?"
            )
        X_raw = X_raw[list(expected_columns)]

    # Cadence sanity check. The model was trained on weekly bars (or
    # daily, depending on cadence in the preprocessing dict). Forecasts
    # become miscalibrated if the caller mixes resolutions because the
    # AR(p) backbone's σ scales with cadence. We warn rather than raise
    # because pandas can't always infer a clean freq on small windows.
    cadence = preprocessing_state.get("cadence", "weekly")
    if len(X_raw) >= 4:
        gaps = X_raw.index.to_series().diff().dt.days.dropna()
        median_gap_days = gaps.median()

        # Weekly bars typically show median gap = 7 days (Sunday-to-Sunday).
        # Daily business-day bars show median gap = 1 day. The thresholds
        # below catch obvious mismatches without false-positiving on
        # holiday gaps.
        if cadence == "daily" and median_gap_days > 3:
            warnings.warn(
                f"x_latest median gap {median_gap_days}d looks weekly, but model "
                f"was trained at cadence='daily'; forecasts may be miscalibrated.",
                UserWarning,
                stacklevel=2,
            )
        elif cadence == "weekly" and median_gap_days < 4:
            warnings.warn(
                f"x_latest median gap {median_gap_days}d looks daily, but model "
                f"was trained at cadence='weekly'.",
                UserWarning,
                stacklevel=2,
            )

    return X_raw
