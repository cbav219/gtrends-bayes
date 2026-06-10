"""End-to-end smoke tests for the BSTS rpy2 wrapper.

These hit a live R subprocess via rpy2 — slower than the rest of the suite
(~10s total) but absolutely necessary to confirm the binding still works after
any R / bsts / rpy2 upgrade. If R / bsts isn't installed, the whole module
skips cleanly so CI without R can still run the rest of the suite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# rpy2 prints an ABI-fallback warning on import — irrelevant for tests.
warnings.filterwarnings("ignore")

pytest.importorskip("rpy2", reason="rpy2 not installed — skipping BSTS smoke tests")

# Importing the wrapper triggers `_init_r` lazily on first use, so we don't
# pay the rpy2 cost during test collection. Anything below this point requires
# a working R + bsts install.

from gtrends_bayes.models.bsts import BSTS, reset_r_models  # noqa: E402


@pytest.fixture(scope="module")
def fitted_bsts():
    """One BSTS fit shared across the BSTS-with-regression tests."""
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2018-01-07", periods=n, freq="W-SUN")
    x_signal = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    y_arr = np.cumsum(0.05 * rng.standard_normal(n)) + 0.6 * x_signal + 0.05 * rng.standard_normal(n)
    y = pd.Series(y_arr, index=idx, name="y")
    X = pd.DataFrame({"signal": x_signal, "noise": x_noise}, index=idx)

    model = BSTS(n_seasons=52, expected_predictors=2, niter=600, burn=60, seed=1)
    model.fit(y, X)
    yield model
    reset_r_models()


def test_bsts_constructs_with_defaults():
    m = BSTS()
    assert m.niter == 3000
    assert m.burn == 300
    assert not m._fitted
    assert "fitted=False" in repr(m)


def test_bsts_fit_marks_has_regression(fitted_bsts):
    assert fitted_bsts._fitted is True
    assert fitted_bsts._has_regression is True


def test_bsts_recovers_signal_and_drops_noise(fitted_bsts):
    """Spike-and-slab MUST give the signal column high inclusion prob, noise low."""
    probs = fitted_bsts.inclusion_probabilities()
    assert "signal" in probs.index
    assert "noise" in probs.index
    assert probs["signal"] > 0.90, f"signal incl prob {probs['signal']:.3f} should be > 0.9"
    assert probs["noise"] < 0.30, f"noise incl prob {probs['noise']:.3f} should be < 0.3"
    # Intercept hidden by default.
    assert "(Intercept)" not in probs.index


def test_bsts_coefficient_summary_columns(fitted_bsts):
    summary = fitted_bsts.coefficient_summary()
    assert {"inclusion_prob", "mean_when_included", "sd_when_included",
            "sign_consistency"}.issubset(set(summary.columns))
    # signal coefficient should be positive (true coef = 0.6).
    assert summary.loc["signal", "mean_when_included"] > 0


def test_bsts_forecast_shape(fitted_bsts):
    rng = np.random.default_rng(2)
    X_future = pd.DataFrame({"signal": rng.standard_normal(4),
                             "noise": rng.standard_normal(4)})
    fc = fitted_bsts.forecast(horizon=4, X_future=X_future)
    # 600 iter - 60 burn = 540 kept draws.
    assert fc.shape == (540, 4)


def test_bsts_forecast_requires_x_future_when_regression(fitted_bsts):
    with pytest.raises(ValueError, match="X_future is required"):
        fitted_bsts.forecast(horizon=4)


def test_bsts_component_bands_returns_trend_and_seasonal(fitted_bsts):
    bands = fitted_bsts.component_bands()
    # bsts names the trend component "trend" and the seasonal one "seasonal.<n>.<idx>".
    assert "trend" in bands
    assert any(name.startswith("seasonal") for name in bands)
    for name, df in bands.items():
        assert {"q_low", "q_med", "q_high"}.issubset(df.columns)
        # in-sample length matches y's length (240).
        assert len(df) == 240


def test_bsts_to_arviz_returns_datatree(fitted_bsts):
    """ArviZ 1.0 swapped InferenceData for xarray DataTree — verify the new shape."""
    idata = fitted_bsts.to_arviz()
    # Must contain a "posterior" subgroup with sigma_obs + beta variables.
    assert "posterior" in idata.children
    posterior = idata["posterior"]
    assert "sigma_obs" in posterior.data_vars
    assert "beta" in posterior.data_vars
    assert "inclusion" in posterior.data_vars


def test_bsts_pure_structural_no_x():
    """Without X, fit should still complete and inclusion_probabilities is empty."""
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    y = pd.Series(np.cumsum(rng.standard_normal(n) * 0.2), index=idx, name="y")
    m = BSTS(n_seasons=52, expected_predictors=1, niter=300, burn=30, seed=4)
    m.fit(y, X=None)
    assert m._has_regression is False
    assert m.inclusion_probabilities().empty
    reset_r_models()


def test_bsts_unfit_methods_raise():
    m = BSTS()
    with pytest.raises(RuntimeError):
        m.inclusion_probabilities()
    with pytest.raises(RuntimeError):
        m.coefficient_summary()
    with pytest.raises(RuntimeError):
        m.forecast(horizon=1)
