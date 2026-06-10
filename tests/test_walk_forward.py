"""Tests for backtest.walk_forward.WalkForward (Phase 6)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

from gtrends_bayes.backtest.walk_forward import WalkForward
from gtrends_bayes.models.baseline import AR_p, NaiveRW


@pytest.fixture
def synthetic_xy(weekly_index, rng):
    """Drift + noise y with a small but real signal in X."""
    n = len(weekly_index)
    x_signal = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    y = np.cumsum(0.05 * rng.standard_normal(n)) + 0.4 * x_signal
    return (
        pd.DataFrame({"signal": x_signal, "noise": x_noise}, index=weekly_index),
        pd.Series(y, index=weekly_index, name="y"),
    )


def test_walk_forward_runs_with_naive_rw(synthetic_xy):
    """NaiveRW has no regression component, so walk-forward must work without X_future."""
    X, y = synthetic_xy
    # Empty X column set forces the no-regression path.
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, step=1, horizon=1, refit_every=10, publication_lag=1)
    out = wf.run(NaiveRW, X_empty, y, n_draws=200)
    assert isinstance(out, pd.DataFrame)
    assert {"y_true", "y_pred_mean", "q500", "refit"}.issubset(out.columns)
    assert len(out) > 100


def test_walk_forward_runs_with_ar_p(synthetic_xy):
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, step=1, horizon=1, refit_every=10, publication_lag=1)
    out = wf.run(lambda: AR_p(p=4), X_empty, y, n_draws=200)
    assert len(out) > 100
    # With a slightly-predictable series, AR(p)'s y_pred shouldn't be wildly off.
    err_std = (out["y_true"] - out["y_pred_mean"]).std()
    assert err_std < y.std() * 2.0  # very loose sanity check


def test_walk_forward_publication_lag_respected(synthetic_xy):
    """At step t, the latest training y must be at index t - publication_lag - 1."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    lag = 3
    wf = WalkForward(train_window=120, step=1, horizon=1, refit_every=10, publication_lag=lag)
    out = wf.run(NaiveRW, X_empty, y, n_draws=10)
    # First forecast date should be at least train_window + publication_lag from start.
    first_forecast = out.index[0]
    pos = X.index.get_loc(first_forecast)
    assert pos >= 120 + lag


def test_walk_forward_refit_marks_first_step():
    """The very first step is always a refit; subsequent ones only when scheduled."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    y = pd.Series(np.cumsum(rng.standard_normal(n)), index=idx, name="y")
    X = pd.DataFrame(index=idx)
    wf = WalkForward(train_window=80, step=1, horizon=1, refit_every=10, publication_lag=1)
    out = wf.run(NaiveRW, X, y, n_draws=10)
    assert out["refit"].iloc[0] == 1
    # Number of refits: every 10 steps + 1 initial.
    n_refits = int(out["refit"].sum())
    expected = (len(out) + 9) // 10
    assert n_refits == pytest.approx(expected, abs=1)


def test_walk_forward_rejects_mismatched_index():
    idx_a = pd.date_range("2020-01-05", periods=200, freq="W-SUN")
    idx_b = pd.date_range("2021-01-03", periods=200, freq="W-SUN")
    X = pd.DataFrame(index=idx_a)
    y = pd.Series(0.0, index=idx_b, name="y")
    wf = WalkForward(train_window=10)
    with pytest.raises(ValueError, match="same index"):
        wf.run(NaiveRW, X, y)


def test_walk_forward_rejects_too_short_history():
    idx = pd.date_range("2020-01-05", periods=20, freq="W-SUN")
    X = pd.DataFrame(index=idx)
    y = pd.Series(0.0, index=idx, name="y")
    wf = WalkForward(train_window=100, publication_lag=1)
    with pytest.raises(ValueError, match="at least"):
        wf.run(NaiveRW, X, y)
