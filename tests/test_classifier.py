"""Tests for DirectionalForecaster (Phase B)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

pytest.importorskip("rpy2", reason="rpy2 not installed — skipping classifier tests")

from gtrends_bayes.models.bsts import BSTS, reset_r_models  # noqa: E402
from gtrends_bayes.models.classifier import DirectionalForecaster  # noqa: E402


@pytest.fixture(scope="module")
def fitted_bsts():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2018-01-07", periods=n, freq="W-SUN")
    x_signal = rng.standard_normal(n)
    y_arr = np.cumsum(0.05 * rng.standard_normal(n)) + 0.6 * x_signal
    y = pd.Series(y_arr, index=idx, name="y")
    X = pd.DataFrame({"signal": x_signal}, index=idx)
    m = BSTS(n_seasons=52, expected_predictors=1, niter=400, burn=40, seed=1).fit(y, X)
    yield m
    reset_r_models()


def test_directional_forecaster_requires_fitted_bsts():
    m = BSTS()
    with pytest.raises(ValueError, match="BSTS must be fit"):
        DirectionalForecaster(m)


def test_predict_proba_in_unit_interval(fitted_bsts):
    df = DirectionalForecaster(fitted_bsts)
    rng = np.random.default_rng(2)
    X_future = pd.DataFrame({"signal": rng.standard_normal(4)})
    probs = df.predict_proba(X_future, y_baseline=0.0)
    assert isinstance(probs, pd.Series)
    assert len(probs) == 4
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_predict_proba_increase_high_baseline_low_low_baseline_high(fitted_bsts):
    """P(y_forecast > baseline) should be near 0 for very high baseline, near 1 for very low."""
    df = DirectionalForecaster(fitted_bsts)
    rng = np.random.default_rng(2)
    X_future = pd.DataFrame({"signal": rng.standard_normal(4)})
    p_high = df.predict_proba(X_future, y_baseline=1e6, direction="increase")
    p_low = df.predict_proba(X_future, y_baseline=-1e6, direction="increase")
    assert p_high.max() < 0.05
    assert p_low.min() > 0.95


def test_predict_proba_decrease_inverts(fitted_bsts):
    df = DirectionalForecaster(fitted_bsts)
    rng = np.random.default_rng(2)
    X_future = pd.DataFrame({"signal": rng.standard_normal(4)})
    p_inc = df.predict_proba(X_future, y_baseline=0.0, direction="increase")
    p_dec = df.predict_proba(X_future, y_baseline=0.0, direction="decrease")
    # P(>baseline) + P(<baseline) ≈ 1 (modulo P(==baseline) which is ~0 for continuous draws)
    assert (p_inc + p_dec - 1.0).abs().max() < 0.05


def test_predict_proba_unknown_direction_raises(fitted_bsts):
    df = DirectionalForecaster(fitted_bsts)
    X_future = pd.DataFrame({"signal": [0.0]})
    with pytest.raises(ValueError, match="unknown direction"):
        df.predict_proba(X_future, y_baseline=0.0, direction="lateral")  # type: ignore[arg-type]


def test_predict_proba_threshold_three_directions(fitted_bsts):
    df = DirectionalForecaster(fitted_bsts)
    rng = np.random.default_rng(3)
    X_future = pd.DataFrame({"signal": rng.standard_normal(2)})
    p_above = df.predict_proba_threshold(X_future, y_baseline=0.0, threshold=0.5, direction="above")
    p_below = df.predict_proba_threshold(X_future, y_baseline=0.0, threshold=0.5, direction="below")
    p_either = df.predict_proba_threshold(X_future, y_baseline=0.0, threshold=0.5, direction="either")
    # P(either) should equal P(above) + P(below) since the events are disjoint.
    assert (p_either - (p_above + p_below)).abs().max() < 0.02
