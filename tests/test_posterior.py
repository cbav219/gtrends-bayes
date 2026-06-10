"""Tests for models.posterior helpers (forecast intervals, inclusion table, long decomposition)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.models.posterior import (
    decompose_to_long,
    forecast_intervals,
    inclusion_table,
)


# ---- forecast_intervals ----------------------------------------------------

def test_forecast_intervals_returns_quantile_columns():
    rng = np.random.default_rng(0)
    fc = pd.DataFrame(rng.normal(100, 5, size=(1000, 4)))
    bands = forecast_intervals(fc, levels=(0.8, 0.95))
    # Expect q050 (median) plus q100, q900 (80%) and q025, q975 (95%).
    assert "q500" in bands.columns
    assert {"q100", "q900", "q025", "q975"}.issubset(set(bands.columns))
    assert list(bands.index) == [1, 2, 3, 4]


def test_forecast_intervals_empty_returns_empty():
    out = forecast_intervals(pd.DataFrame(), levels=(0.8,))
    assert out.empty


def test_forecast_intervals_quantiles_ordered():
    rng = np.random.default_rng(0)
    fc = pd.DataFrame(rng.normal(0, 1, size=(2000, 1)))
    bands = forecast_intervals(fc, levels=(0.5, 0.8, 0.95))
    row = bands.iloc[0].sort_index()  # sort by quantile name to enforce order
    cols = sorted(bands.columns)      # q025, q100, q250, q500, q750, q900, q975
    vals = [row[c] for c in cols]
    assert vals == sorted(vals), "quantile values should be monotone non-decreasing"


# ---- decompose_to_long & inclusion_table -----------------------------------

class _MockBSTS:
    """Minimal duck-typed BSTS for posterior-helper tests (avoids R round-trip)."""

    def __init__(self, summary: pd.DataFrame, bands: dict[str, pd.DataFrame]):
        self._summary = summary
        self._bands = bands

    def coefficient_summary(self):
        return self._summary

    def component_bands(self):
        return self._bands


def test_inclusion_table_adds_sign_column():
    summary = pd.DataFrame({
        "inclusion_prob": [0.95, 0.50, 0.05],
        "mean_when_included": [0.4, -0.2, 0.0],
        "sd_when_included": [0.05, 0.10, np.nan],
        "sign_consistency": [1.0, 1.0, np.nan],
    }, index=pd.Index(["a", "b", "c"], name="predictor"))
    out = inclusion_table(_MockBSTS(summary, {}))
    assert "sign" in out.columns
    assert out.loc["a", "sign"] == 1
    assert out.loc["b", "sign"] == -1
    assert out.loc["c", "sign"] == 0


def test_decompose_to_long_format():
    idx = pd.date_range("2020-01-05", periods=4, freq="W-SUN")
    bands = {
        "trend": pd.DataFrame({"q_low": [0, 1, 2, 3], "q_med": [1, 2, 3, 4],
                               "q_high": [2, 3, 4, 5]}, index=idx),
        "seasonal": pd.DataFrame({"q_low": [-1, 0, 1, 0], "q_med": [0, 1, 0, -1],
                                  "q_high": [1, 2, 1, 0]}, index=idx),
    }
    long_df = decompose_to_long(_MockBSTS(pd.DataFrame(), bands))
    assert set(long_df.columns) == {"date", "component", "quantile", "value"}
    # 2 components × 3 quantiles × 4 dates = 24 rows.
    assert len(long_df) == 2 * 3 * 4
    assert set(long_df["component"].unique()) == {"trend", "seasonal"}
    assert set(long_df["quantile"].unique()) == {"q_low", "q_med", "q_high"}


def test_decompose_to_long_empty_bands():
    out = decompose_to_long(_MockBSTS(pd.DataFrame(), {}))
    assert out.empty
    assert list(out.columns) == ["date", "component", "quantile", "value"]
