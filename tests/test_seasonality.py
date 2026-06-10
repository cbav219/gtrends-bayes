"""Tests for preprocessing.seasonality (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gtrends_bayes.preprocessing.seasonality import transform_by_class, yoy_log_diff


def test_yoy_log_diff_kills_annual_seasonality(synthetic_svi):
    """The 52-week sinusoid in the fixture should disappear under YoY log-differencing.

    The synthetic SVI has: linear drift + 52-week sinusoid + noise. After
    log-transform and YoY differencing, the seasonal component cancels (same
    phase 52 weeks apart), leaving only the year-over-year drift change + noise.
    """
    log_svi = np.log(synthetic_svi.to_frame())
    diffed = yoy_log_diff(log_svi, periods_per_year=52, weighted_neighbor=False)
    # First 52 entries are NaN (no prior-year reference).
    assert diffed.iloc[:52].isna().all().all()
    valid = diffed.dropna().iloc[:, 0]
    # YoY drift on the synthetic fixture is small; std should be a small
    # fraction of the original log-SVI std (which is dominated by seasonal +
    # drift). The remaining variance is ~2x noise (independent draws).
    assert valid.std() < log_svi.iloc[:, 0].std() * 0.5


def test_yoy_log_diff_simple_form_matches_pandas_shift():
    df = pd.DataFrame({
        "x": np.arange(60.0, dtype=float),
    }, index=pd.date_range("2020-01-05", periods=60, freq="W-SUN"))
    out = yoy_log_diff(df, periods_per_year=52, weighted_neighbor=False)
    # All non-NaN values should equal 52 (since x_t - x_{t-52} = 52 for arange).
    assert (out.dropna().iloc[:, 0] == 52).all()


def test_yoy_log_diff_weighted_uses_three_lags():
    """Weighted variant must look at lags 51, 52, 53 — needs ≥54 rows of input."""
    n = 60
    df = pd.DataFrame({"x": np.arange(n, dtype=float)},
                      index=pd.date_range("2020-01-05", periods=n, freq="W-SUN"))
    out = yoy_log_diff(df, periods_per_year=52, weighted_neighbor=True)
    # First 53 rows are NaN (need t-53 to exist).
    assert out.iloc[:53].isna().all().all()
    # For arange, x_t - weighted_avg(x_{t-51}, x_{t-52}, x_{t-53}) = 52.
    valid = out.dropna().iloc[:, 0]
    assert np.allclose(valid.values, 52.0)


def test_transform_by_class_passes_topics_through(synthetic_svi):
    """Topic columns should be returned unchanged (log-levels, not differenced)."""
    log_svi = np.log(synthetic_svi.to_frame().rename(columns={"synthetic": "topic_X"}))
    out = transform_by_class(log_svi, classes={"topic_X": "topic"}, periods_per_year=52)
    pd.testing.assert_frame_equal(out, log_svi)


def test_transform_by_class_diffs_categories():
    n = 60
    base = np.arange(n, dtype=float)
    df = pd.DataFrame({"cat_A": base, "topic_B": base},
                      index=pd.date_range("2020-01-05", periods=n, freq="W-SUN"))
    out = transform_by_class(df,
                              classes={"cat_A": "category", "topic_B": "topic"},
                              periods_per_year=52, weighted_neighbor=False)
    # Category column should be YoY-differenced (constant 52 after lag).
    assert (out["cat_A"].dropna() == 52).all()
    # Topic column should be unchanged.
    pd.testing.assert_series_equal(out["topic_B"], df["topic_B"], check_names=False)


def test_transform_by_class_preserves_column_order():
    n = 60
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in ["b", "a", "c"]},
                      index=pd.date_range("2020-01-05", periods=n, freq="W-SUN"))
    out = transform_by_class(df, classes={"b": "topic", "a": "category", "c": "topic"})
    assert list(out.columns) == ["b", "a", "c"]
