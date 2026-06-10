"""Tests for backtest.metrics (Phase 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.backtest.metrics import (
    directional_hit_rate,
    mae,
    posterior_coverage,
    rmse,
    rmse_ratio,
    standardized_rmse,
)


# ---- rmse / mae ------------------------------------------------------------

def test_rmse_zero_when_perfect():
    s = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    assert rmse(s, s) == 0.0


def test_rmse_known_value():
    y = pd.Series([0.0, 0.0, 0.0])
    p = pd.Series([1.0, -1.0, 1.0])
    # MSE = (1+1+1)/3 = 1; RMSE = 1
    assert rmse(y, p) == pytest.approx(1.0)


def test_mae_known_value():
    y = pd.Series([0.0, 0.0, 0.0])
    p = pd.Series([1.0, -2.0, 3.0])
    # MAE = (1+2+3)/3 = 2
    assert mae(y, p) == pytest.approx(2.0)


def test_rmse_handles_index_mismatch():
    y = pd.Series([0.0, 1.0, 2.0], index=[0, 1, 2])
    p = pd.Series([0.0, 1.0], index=[0, 1])  # missing index 2
    # Common indices = {0, 1}; both errors = 0; RMSE = 0
    assert rmse(y, p) == 0.0


# ---- standardized_rmse / rmse_ratio ----------------------------------------

def test_standardized_rmse_unit_when_pred_is_mean():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, 1000))
    p = pd.Series(np.full(1000, y.mean()))
    # RMSE ≈ std(y); standardized_rmse ≈ 1
    assert standardized_rmse(y, p) == pytest.approx(1.0, rel=0.05)


def test_rmse_ratio_baseline_better_returns_above_one():
    y = pd.Series([0.0, 0.0, 0.0])
    p_bad = pd.Series([2.0, -2.0, 2.0])
    p_good = pd.Series([0.5, -0.5, 0.5])
    assert rmse_ratio(y, p_bad, p_good) > 1.0


# ---- directional_hit_rate --------------------------------------------------

def test_directional_hit_rate_perfect_when_signs_match():
    y = pd.Series([1.0, 2.0, 1.5, 3.0])
    p = pd.Series([1.0, 1.5, 1.4, 2.0])  # same direction at each step
    assert directional_hit_rate(y, p) == pytest.approx(1.0)


def test_directional_hit_rate_zero_when_signs_opposite():
    y = pd.Series([1.0, 2.0, 1.5, 3.0])  # diffs: +, -, +
    p = pd.Series([1.0, 0.0, 1.0, 0.5])  # diffs: -, +, -
    assert directional_hit_rate(y, p) == pytest.approx(0.0)


# ---- posterior_coverage ----------------------------------------------------

def test_posterior_coverage_perfect_when_intervals_cover_all():
    idx = pd.date_range("2020-01-05", periods=4, freq="W-SUN")
    y = pd.Series([10.0, 11.0, 9.5, 10.5], index=idx, name="y")
    bands = pd.DataFrame({
        "q025": [0.0, 0.0, 0.0, 0.0],
        "q975": [100.0, 100.0, 100.0, 100.0],
    }, index=idx)
    cov = posterior_coverage(y, bands, levels=(0.95,))
    assert cov[0.95] == 1.0


def test_posterior_coverage_zero_when_intervals_miss():
    idx = pd.date_range("2020-01-05", periods=4, freq="W-SUN")
    y = pd.Series([10.0, 11.0, 9.5, 10.5], index=idx, name="y")
    bands = pd.DataFrame({
        "q025": [20.0, 20.0, 20.0, 20.0],   # all too high
        "q975": [30.0, 30.0, 30.0, 30.0],
    }, index=idx)
    cov = posterior_coverage(y, bands, levels=(0.95,))
    assert cov[0.95] == 0.0


def test_posterior_coverage_missing_bands_returns_nan():
    idx = pd.date_range("2020-01-05", periods=4, freq="W-SUN")
    y = pd.Series([10.0, 11.0, 9.5, 10.5], index=idx, name="y")
    bands = pd.DataFrame({"q025": [0.0]*4}, index=idx)  # missing q975
    cov = posterior_coverage(y, bands, levels=(0.95,))
    assert np.isnan(cov[0.95])
