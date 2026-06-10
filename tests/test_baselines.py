"""Tests for models.baseline (Phase 5 baselines for backtest comparison)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# AutoReg emits ConvergenceWarning on tiny test fixtures — drown it out.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

from gtrends_bayes.models.baseline import AR_p, AR_VIX, NaiveRW


@pytest.fixture
def random_walk_series():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    return pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx, name="y")


@pytest.fixture
def vix_logdiff(random_walk_series):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"vix_logdiff": rng.normal(0, 0.05, len(random_walk_series))},
        index=random_walk_series.index,
    )


# ---- NaiveRW ---------------------------------------------------------------

def test_naive_rw_forecast_mean_centered_on_last_value(random_walk_series):
    m = NaiveRW(seed=0).fit(random_walk_series)
    fc = m.forecast(horizon=4, n_draws=2000)
    last = float(random_walk_series.iloc[-1])
    # 1-step mean ≈ last value (within sampling noise).
    assert abs(fc.iloc[:, 0].mean() - last) < 1.0


def test_naive_rw_intervals_widen_with_horizon(random_walk_series):
    m = NaiveRW(seed=0).fit(random_walk_series)
    fc = m.forecast(horizon=8, n_draws=2000)
    stds = fc.std(axis=0).values
    # Random walk: std grows monotonically with horizon (modulo Monte-Carlo noise).
    assert stds[-1] > stds[0] * 1.5


def test_naive_rw_unfit_raises():
    with pytest.raises(RuntimeError):
        NaiveRW().forecast(horizon=1, n_draws=10)


# ---- AR(p) -----------------------------------------------------------------

def test_ar_p_fit_and_forecast_shape(random_walk_series):
    m = AR_p(p=4, seed=0).fit(random_walk_series)
    fc = m.forecast(horizon=4, n_draws=500)
    assert fc.shape == (500, 4)


def test_ar_p_unfit_raises():
    with pytest.raises(RuntimeError):
        AR_p(p=2).forecast(horizon=1)


# ---- AR(p) + VIX -----------------------------------------------------------

def test_ar_vix_requires_x_at_fit(random_walk_series):
    with pytest.raises(ValueError):
        AR_VIX(p=2).fit(random_walk_series, X=None)  # type: ignore[arg-type]


def test_ar_vix_requires_x_future(random_walk_series, vix_logdiff):
    m = AR_VIX(p=4, seed=0).fit(random_walk_series, vix_logdiff)
    with pytest.raises(ValueError):
        m.forecast(horizon=4, X_future=None)  # type: ignore[arg-type]


def test_ar_vix_x_future_must_match_horizon(random_walk_series, vix_logdiff):
    m = AR_VIX(p=4, seed=0).fit(random_walk_series, vix_logdiff)
    bad = pd.DataFrame({"vix_logdiff": [0.01, 0.02]})  # 2 rows but horizon=4
    with pytest.raises(ValueError):
        m.forecast(horizon=4, X_future=bad)


def test_ar_vix_forecast_shape(random_walk_series, vix_logdiff):
    m = AR_VIX(p=4, seed=0).fit(random_walk_series, vix_logdiff)
    X_future = pd.DataFrame({"vix_logdiff": [0.01, -0.02, 0.03, 0.0]})
    fc = m.forecast(horizon=4, X_future=X_future, n_draws=200)
    assert fc.shape == (200, 4)
