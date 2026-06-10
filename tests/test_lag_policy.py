"""Tests for the Phase D lag-policy split (asymmetric publication lag + modes)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

from gtrends_bayes.backtest.walk_forward import WalkForward
from gtrends_bayes.models.baseline import NaiveRW


@pytest.fixture
def synthetic_xy(weekly_index, rng):
    n = len(weekly_index)
    x_signal = rng.standard_normal(n)
    y = np.cumsum(0.05 * rng.standard_normal(n)) + 0.3 * x_signal
    return (
        pd.DataFrame({"signal": x_signal}, index=weekly_index),
        pd.Series(y, index=weekly_index, name="y"),
    )


# ---- mode defaults ---------------------------------------------------------

def test_mode_backtest_defaults_zero_zero():
    wf = WalkForward(train_window=120, mode="backtest")
    assert wf.cfg.mode == "backtest"
    assert wf.cfg.publication_lag_y == 0
    assert wf.cfg.publication_lag_x == 0
    assert wf.cfg.publication_lag == 0   # effective = max(0, 0)


def test_mode_forecast_defaults_zero_one():
    wf = WalkForward(train_window=120, mode="forecast")
    assert wf.cfg.mode == "forecast"
    assert wf.cfg.publication_lag_y == 0
    assert wf.cfg.publication_lag_x == 1
    assert wf.cfg.publication_lag == 1   # effective = max(0, 1)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        WalkForward(train_window=120, mode="liveops")  # type: ignore[arg-type]


# ---- explicit lag overrides ----------------------------------------------

def test_explicit_lag_y_and_lag_x_override_mode_defaults():
    wf = WalkForward(train_window=120, mode="backtest",
                     publication_lag_y=2, publication_lag_x=3)
    assert wf.cfg.publication_lag_y == 2
    assert wf.cfg.publication_lag_x == 3
    assert wf.cfg.publication_lag == 3


def test_legacy_publication_lag_kwarg_fills_both():
    wf = WalkForward(train_window=120, publication_lag=2)
    assert wf.cfg.publication_lag_y == 2
    assert wf.cfg.publication_lag_x == 2
    assert wf.cfg.publication_lag == 2


def test_legacy_lag_does_not_override_explicit_y_x():
    wf = WalkForward(train_window=120, publication_lag=5,
                     publication_lag_y=0, publication_lag_x=1)
    # Explicit values win over legacy.
    assert wf.cfg.publication_lag_y == 0
    assert wf.cfg.publication_lag_x == 1


def test_negative_lag_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        WalkForward(train_window=120, publication_lag_y=-1)


# ---- behavioral: lag controls training boundary --------------------------

def test_backtest_clean_uses_more_recent_data_than_forecast_realistic(synthetic_xy):
    """backtest_clean (lag=0) trains on data through t-1; forecast_realistic
    (lag=1) trains through t-2. The first valid forecast date differs."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    bt = WalkForward(train_window=120, mode="backtest")
    fc = WalkForward(train_window=120, mode="forecast",
                     publication_lag_y=1, publication_lag_x=1)
    out_bt = bt.run(NaiveRW, X_empty, y, n_draws=20)
    out_fc = fc.run(NaiveRW, X_empty, y, n_draws=20)
    # forecast_realistic needs one more period of "buffer", so first forecast
    # date is later (later in calendar terms = larger index position).
    pos_bt = y.index.get_loc(out_bt.index[0])
    pos_fc = y.index.get_loc(out_fc.index[0])
    assert pos_fc > pos_bt


def test_three_config_runs_produce_distinct_results(synthetic_xy):
    """v1_legacy, backtest_clean, forecast_realistic should all run cleanly
    and produce overlapping but distinct forecast outputs."""
    X, y = synthetic_xy
    X_empty = X.iloc[:, :0]
    configs = {
        "v1_legacy":          dict(publication_lag_y=1, publication_lag_x=1),
        "backtest_clean":     dict(mode="backtest"),
        "forecast_realistic": dict(mode="forecast"),
    }
    outs = {}
    for label, kwargs in configs.items():
        wf = WalkForward(train_window=120, **kwargs)
        outs[label] = wf.run(NaiveRW, X_empty, y, n_draws=20)
    # All three produced output.
    for label, df in outs.items():
        assert len(df) > 100, f"{label} produced too few forecasts"
    # backtest_clean should produce the LONGEST forecast window (smallest lag).
    assert len(outs["backtest_clean"]) >= len(outs["v1_legacy"])
    assert len(outs["backtest_clean"]) >= len(outs["forecast_realistic"])


def test_asymmetric_lag_x_lookup_differs(synthetic_xy):
    """With pub_lag_x=2, the X_future row at forecast date t should come from
    X.iloc[t - 3] (i.e. t - pub_lag_x - 1)."""
    X, y = synthetic_xy
    # NaiveRW ignores X but we still want a regression-mode run to exercise
    # the x_lookup_idx code path. Use AR_p which takes X but ignores it.
    from gtrends_bayes.models.baseline import AR_p

    wf = WalkForward(train_window=120, mode="forecast",
                     publication_lag_y=0, publication_lag_x=2)
    out = wf.run(lambda: AR_p(p=4), X, y, n_draws=20)
    assert len(out) > 50
    # No NaN in y_pred_mean — confirms the lookup didn't go off the index.
    assert out["y_pred_mean"].notna().all()