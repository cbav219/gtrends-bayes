"""Tests for the multi-horizon WalkForward path (Phase B)."""

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
    n = len(weekly_index)
    x_signal = rng.standard_normal(n)
    y = np.cumsum(0.05 * rng.standard_normal(n)) + 0.4 * x_signal
    return (
        pd.DataFrame({"signal": x_signal}, index=weekly_index),
        pd.Series(y, index=weekly_index, name="y"),
    )


def test_walk_forward_legacy_horizon_unchanged(synthetic_xy):
    """Passing legacy ``horizon=1`` returns single-horizon wide format (back-compat)."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, horizon=1, refit_every=10, publication_lag=1)
    out = wf.run(NaiveRW, X_empty, y, n_draws=100)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "horizon" not in out.columns


def test_walk_forward_horizons_returns_long_format(synthetic_xy):
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, horizons=[1, 2, 4],
                     refit_every=10, publication_lag=1)
    out = wf.run(NaiveRW, X_empty, y, n_draws=100)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names == ["forecast_date", "horizon"]
    # Three horizons, so we should be able to xs each one.
    for h in (1, 2, 4):
        slice_h = out.xs(h, level="horizon")
        assert len(slice_h) > 0


def test_walk_forward_horizons_y_true_matches_index_offset(synthetic_xy):
    """y_true at horizon h must be y at index t+h-1 (the predicted date)."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, horizons=[1, 4], refit_every=10, publication_lag=1)
    out = wf.run(NaiveRW, X_empty, y, n_draws=50)
    # For h=1, forecast_date should match y's index where y_true equals y.
    h1 = out.xs(1, level="horizon")
    for forecast_date, row in h1.head(5).iterrows():
        assert y.loc[forecast_date] == pytest.approx(row["y_true"])


def test_walk_forward_rejects_both_horizon_kwargs():
    with pytest.raises(ValueError, match="either horizon .* or horizons"):
        WalkForward(horizon=1, horizons=[1, 2])


def test_walk_forward_rejects_zero_horizon():
    with pytest.raises(ValueError, match="must be >= 1"):
        WalkForward(horizons=[0, 1])


def test_walk_forward_rejects_empty_horizons():
    with pytest.raises(ValueError, match="at least one entry"):
        WalkForward(horizons=[])


def test_walk_forward_long_horizons_need_more_history(synthetic_xy):
    """Forecasts for h=13 need t+12 to exist, so test window is shorter."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf_h1 = WalkForward(train_window=120, horizons=[1], refit_every=10, publication_lag=1)
    wf_h13 = WalkForward(train_window=120, horizons=[13], refit_every=10, publication_lag=1)
    out_h1 = wf_h1.run(NaiveRW, X_empty, y, n_draws=20)
    out_h13 = wf_h13.run(NaiveRW, X_empty, y, n_draws=20)
    assert len(out_h13) == len(out_h1) - 12, "long horizons drop the last (h-1) steps"


def test_walk_forward_multi_horizon_with_ar_p(synthetic_xy):
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, horizons=[1, 4, 13],
                     refit_every=10, publication_lag=1)
    out = wf.run(lambda: AR_p(p=4), X_empty, y, n_draws=100)
    assert isinstance(out.index, pd.MultiIndex)
    # Each horizon's predictions should be different (AR(p) has horizon-dependent dynamics).
    h1 = out.xs(1, level="horizon")["y_pred_mean"]
    h13 = out.xs(13, level="horizon")["y_pred_mean"]
    common = h1.index.intersection(h13.index)
    if len(common) > 5:
        assert (h1.loc[common] - h13.loc[common]).std() > 0


def test_walk_forward_mode_param_accepted(synthetic_xy):
    """mode kwarg lands in Phase B but doesn't change behavior until Phase D."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    wf = WalkForward(train_window=120, horizons=[1], refit_every=10,
                     publication_lag=1, mode="forecast")
    out = wf.run(NaiveRW, X_empty, y, n_draws=20)
    assert wf.cfg.mode == "forecast"
    assert len(out) > 0
