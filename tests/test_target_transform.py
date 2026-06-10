"""Round-trip + edge-case tests for preprocessing/target_transform.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.preprocessing.target_transform import TargetTransform


@pytest.fixture
def oas_levels() -> pd.Series:
    """An OAS-shaped series: monotonic-ish positive values with one big spike."""
    idx = pd.bdate_range("2020-01-02", periods=100)
    rng = np.random.default_rng(0)
    base = 350.0 + np.cumsum(rng.normal(0, 5, size=100))  # bps random walk
    # Inject a COVID-like spike halfway through.
    base[50:55] += np.array([100, 250, 400, 350, 280])
    return pd.Series(np.clip(base, 50, None), index=idx, name="HY_OAS")


@pytest.mark.parametrize("kind", ["levels", "diff", "log_diff"])
def test_round_trip_recovers_levels(oas_levels: pd.Series, kind: str):
    """transform → inverse_transform should reproduce the original level series."""
    transform = TargetTransform(kind=kind)
    transformed = transform.fit_transform(oas_levels)
    # For diff / log_diff, the first observation is NaN. Reconstruct from t=1.
    if kind == "levels":
        recovered = transform.inverse_transform(transformed)
        pd.testing.assert_series_equal(recovered, oas_levels, check_names=False)
        return

    # Use the anchor at t=0 (start of the series, NOT t=-1 the fit captured).
    # We're reconstructing the forward path from t=1..T given y_0.
    anchor = float(oas_levels.iloc[0])
    forward_path = transformed.iloc[1:]  # drop the NaN at t=0
    recovered = transform.inverse_transform(forward_path, last_level=anchor)
    # Compare to oas_levels[1:].
    target = oas_levels.iloc[1:]
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(target),
                               rtol=1e-10, atol=1e-8)


def test_diff_transform_has_nan_first(oas_levels: pd.Series):
    transform = TargetTransform("diff")
    out = transform.fit_transform(oas_levels)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(oas_levels.iloc[1] - oas_levels.iloc[0])


def test_log_diff_rejects_non_positive():
    """log_diff requires strictly positive levels."""
    bad = pd.Series([100.0, 0.0, 50.0], index=pd.bdate_range("2024-01-01", periods=3))
    transform = TargetTransform("log_diff")
    with pytest.raises(ValueError, match="strictly positive"):
        transform.fit_transform(bad)


def test_inverse_transform_scalar(oas_levels: pd.Series):
    transform = TargetTransform("diff").fit(oas_levels)
    # A single Δ of +12 bps should produce last_level + 12.
    assert transform.inverse_transform(12.0) == pytest.approx(oas_levels.iloc[-1] + 12.0)
    log_t = TargetTransform("log_diff").fit(oas_levels)
    # A log-return of 0 should reproduce the anchor.
    assert log_t.inverse_transform(0.0) == pytest.approx(oas_levels.iloc[-1])


def test_inverse_transform_dataframe_paths(oas_levels: pd.Series):
    """Posterior draws come as a DataFrame (n_horizon × n_draws). Inverse should preserve shape."""
    transform = TargetTransform("diff").fit(oas_levels)
    n_horizon, n_draws = 4, 50
    draws = pd.DataFrame(
        np.random.default_rng(0).normal(0, 3, size=(n_horizon, n_draws)),
        index=pd.bdate_range(oas_levels.index[-1] + pd.Timedelta(days=1), periods=n_horizon),
    )
    out = transform.inverse_transform(draws)
    assert out.shape == draws.shape
    # The first horizon's value should be anchor + draw.
    expected_h1 = oas_levels.iloc[-1] + draws.iloc[0, :]
    np.testing.assert_allclose(out.iloc[0, :].values, expected_h1.values, rtol=1e-12)
    # Last horizon's value should be anchor + sum-of-draws-down-the-column.
    expected_h4 = oas_levels.iloc[-1] + draws.cumsum(axis=0).iloc[-1, :]
    np.testing.assert_allclose(out.iloc[-1, :].values, expected_h4.values, rtol=1e-12)


def test_inverse_transform_requires_anchor():
    transform = TargetTransform("diff")  # not fit, no last_level
    with pytest.raises(RuntimeError, match="anchor"):
        transform.inverse_transform(1.0)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        TargetTransform(kind="square_root")  # type: ignore[arg-type]


def test_levels_inverse_is_identity(oas_levels: pd.Series):
    transform = TargetTransform("levels").fit(oas_levels)
    # No anchor needed, just identity.
    assert transform.inverse_transform(oas_levels) is oas_levels or \
        (transform.inverse_transform(oas_levels) == oas_levels).all()
