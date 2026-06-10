"""Tests for models.stacked_residual.StackedResidualModel (Phase C.1)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

pytest.importorskip("rpy2", reason="rpy2 not installed — skipping stacked-residual tests")

from gtrends_bayes.models.bsts import reset_r_models  # noqa: E402
from gtrends_bayes.models.stacked_residual import StackedResidualModel  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_xy():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2018-01-07", periods=n, freq="W-SUN")
    x_signal = rng.standard_normal(n)
    y_arr = np.cumsum(0.05 * rng.standard_normal(n)) + 0.3 * x_signal
    y = pd.Series(y_arr, index=idx, name="y")
    X = pd.DataFrame({"signal": x_signal, "noise": rng.standard_normal(n)}, index=idx)
    return X, y


@pytest.fixture(scope="module")
def fitted_stacked(synthetic_xy):
    X, y = synthetic_xy
    m = StackedResidualModel(
        ar_p=4,
        bsts_kwargs={"n_seasons": 52, "expected_predictors": 1,
                     "niter": 400, "burn": 40, "seed": 1},
    ).fit(y, X)
    yield m
    reset_r_models()


def test_stacked_residual_requires_x():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(100),
                  index=pd.date_range("2020-01-05", periods=100, freq="W-SUN"))
    with pytest.raises(ValueError, match="X .* is required"):
        StackedResidualModel(ar_p=2).fit(y, X=None)  # type: ignore[arg-type]


def test_stacked_residual_rejects_index_mismatch():
    rng = np.random.default_rng(0)
    idx_y = pd.date_range("2020-01-05", periods=100, freq="W-SUN")
    idx_x = pd.date_range("2021-01-03", periods=100, freq="W-SUN")
    y = pd.Series(rng.standard_normal(100), index=idx_y)
    X = pd.DataFrame({"a": rng.standard_normal(100)}, index=idx_x)
    with pytest.raises(ValueError, match="must share the same index"):
        StackedResidualModel(ar_p=2).fit(y, X)


def test_stacked_residual_fit_completes(fitted_stacked):
    assert fitted_stacked._fitted
    assert fitted_stacked._ar is not None
    assert fitted_stacked._bsts is not None


def test_stacked_residual_forecast_shape(fitted_stacked, synthetic_xy):
    X, _ = synthetic_xy
    rng = np.random.default_rng(2)
    horizon = 4
    X_future = pd.DataFrame({"signal": rng.standard_normal(horizon),
                             "noise": rng.standard_normal(horizon)})
    fc = fitted_stacked.forecast(horizon=horizon, X_future=X_future, n_draws=200)
    assert fc.shape == (200, horizon)


def test_stacked_residual_forecast_mean_equals_ar_plus_bsts_mean(fitted_stacked, synthetic_xy):
    """E[combined] = AR mean + E[BSTS]; subsampling keeps the mean unchanged in expectation."""
    X, _ = synthetic_xy
    rng = np.random.default_rng(2)
    horizon = 4
    X_future = pd.DataFrame({"signal": rng.standard_normal(horizon),
                             "noise": rng.standard_normal(horizon)})
    combined = fitted_stacked.forecast(horizon=horizon, X_future=X_future, n_draws=400)
    ar_mean = fitted_stacked._ar.forecast(horizon=horizon, n_draws=1).mean(axis=0).values
    bsts_fc = fitted_stacked._bsts.forecast(horizon=horizon, X_future=X_future)
    expected_mean = bsts_fc.values.mean(axis=0) + ar_mean
    # Sub-sampling shouldn't shift the mean by more than a few percent of the std.
    bsts_std_per_step = bsts_fc.values.std(axis=0)
    diff = np.abs(combined.mean(axis=0).values - expected_mean)
    tol = np.maximum(0.5 * bsts_std_per_step / np.sqrt(combined.shape[0]), 1e-6)
    assert (diff <= 5 * tol).all(), f"mean drift {diff} exceeded tolerance {5 * tol}"


def test_stacked_residual_attribution_shares_sum_to_one(fitted_stacked):
    attr = fitted_stacked.attribution()
    assert {"y", "ar_pred", "residual_pred", "total_pred",
            "ar_share", "trends_share"}.issubset(attr.columns)
    shares = attr[["ar_share", "trends_share"]].dropna()
    # Magnitude shares should sum to 1 by construction.
    assert np.allclose(shares.sum(axis=1).values, 1.0, atol=1e-9)


def test_stacked_residual_inclusion_probs_pass_through(fitted_stacked):
    probs = fitted_stacked.inclusion_probabilities()
    assert isinstance(probs, pd.Series)
    assert "signal" in probs.index or "noise" in probs.index


def test_stacked_residual_unfit_methods_raise():
    m = StackedResidualModel(ar_p=2)
    with pytest.raises(RuntimeError):
        m.forecast(horizon=1, X_future=pd.DataFrame({"a": [0.0]}))
    with pytest.raises(RuntimeError):
        m.attribution()
    with pytest.raises(RuntimeError):
        m.inclusion_probabilities()
