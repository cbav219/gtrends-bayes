"""Helpers for downstream consumption of BSTS posterior draws."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def inclusion_table(model) -> pd.DataFrame:  # noqa: ANN001 — duck-typed BSTS
    """Inclusion-probability ranking with sign of mean coefficient.

    Thin wrapper around ``BSTS.coefficient_summary`` that adds a ``sign``
    column (+1 / -1 / 0) for plot-color coding and assertion-based tests.
    """
    summary = model.coefficient_summary()
    if summary.empty:
        return summary
    out = summary.copy()
    out["sign"] = np.sign(out["mean_when_included"]).fillna(0).astype(int)
    return out


def forecast_intervals(
    forecast_df: pd.DataFrame,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Quantile bands for a forecast-paths DataFrame ``(n_draws x horizon)``.

    Returns a DataFrame indexed by horizon step with columns named
    ``q005``, ``q025``, ``q050``, ... matching the requested ``levels``. The
    median (q50) is always included.
    """
    if forecast_df.empty:
        return pd.DataFrame()
    quantiles = sorted({0.5} | {0.5 - lvl / 2 for lvl in levels} | {0.5 + lvl / 2 for lvl in levels})
    # round() avoids float-precision tags like q099 instead of q100.
    cols = {f"q{int(round(q * 1000)):03d}": forecast_df.quantile(q, axis=0).values for q in quantiles}
    return pd.DataFrame(cols, index=range(1, forecast_df.shape[1] + 1))


def decompose_to_long(model) -> pd.DataFrame:  # noqa: ANN001
    """Tidy DataFrame: ``date | component | quantile | value``.

    Calls ``BSTS.component_bands()`` and reshapes the per-component DataFrames
    into a single long-format frame suitable for Altair / Seaborn faceting.
    """
    bands = model.component_bands()
    if not bands:
        return pd.DataFrame(columns=["date", "component", "quantile", "value"])
    rows = []
    for component, df in bands.items():
        for col in df.columns:
            piece = df[[col]].rename(columns={col: "value"}).reset_index(names="date")
            piece["component"] = component
            piece["quantile"] = col
            rows.append(piece[["date", "component", "quantile", "value"]])
    return pd.concat(rows, ignore_index=True)
