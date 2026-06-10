"""Tests for preprocessing.bias_removal (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.preprocessing.bias_removal import (
    extract_common_component,
    remove_long_term_drift,
)


@pytest.fixture
def drift_panel(weekly_index, rng):
    """Panel of 4 series sharing a common downward drift + idiosyncratic noise.

    The PC1 of the HP trends should align with the shared drift; subtracting
    it should bring all four series back to a common baseline (their
    idiosyncratic offsets).
    """
    n = len(weekly_index)
    t = np.arange(n)
    drift = -0.025 * t                          # shared downward drift
    cols = {}
    for i, intercept in enumerate([0.0, 0.5, -0.3, 0.2]):
        cols[f"q{i}"] = intercept + drift + rng.normal(0, 0.1, size=n)
    return pd.DataFrame(cols, index=weekly_index)


def test_remove_long_term_drift_kills_known_trend(drift_panel):
    """After de-drifting, no column should retain the steep linear trend."""
    out = remove_long_term_drift(drift_panel, hp_lambda=129600)
    # Slope of a linear regression on the de-drifted series should be tiny.
    t = np.arange(len(out))
    for col in out.columns:
        y = out[col].values
        slope, _intercept = np.polyfit(t, y, 1)
        # Original drift slope was -0.025; post-drift-removal should be ~0.
        assert abs(slope) < 0.005, (
            f"column {col} still has slope {slope:.4f} after drift removal"
        )


def test_remove_long_term_drift_preserves_shape(drift_panel):
    out = remove_long_term_drift(drift_panel)
    assert out.shape == drift_panel.shape
    assert list(out.columns) == list(drift_panel.columns)
    assert out.index.equals(drift_panel.index)


def test_extract_common_component_returns_one_dim_series(drift_panel):
    pc1 = extract_common_component(drift_panel, hp_lambda=129600)
    assert isinstance(pc1, pd.Series)
    assert pc1.index.equals(drift_panel.index)
    # PC1 should track the (negative) shared drift — std should be substantial.
    assert pc1.std() > 0.1


def test_extract_common_component_sign_matches_mean_trend(drift_panel):
    """The returned PC1 must be POSITIVELY correlated with the cross-query mean
    of the original log-SVI. (Sign-flip safeguard inside _common_drift.)"""
    pc1 = extract_common_component(drift_panel, hp_lambda=129600)
    mean_input = drift_panel.mean(axis=1)
    assert pc1.corr(mean_input) > 0.5


def test_remove_long_term_drift_handles_empty_input():
    df = pd.DataFrame()
    out = remove_long_term_drift(df)
    assert out.empty


def test_remove_long_term_drift_short_series_returns_unchanged(weekly_index):
    """Series with too few observations should pass through (logged warning)."""
    short_idx = weekly_index[:5]
    df = pd.DataFrame({"x": np.linspace(50, 40, 5)}, index=short_idx)
    out = remove_long_term_drift(df)
    pd.testing.assert_frame_equal(out, df)
