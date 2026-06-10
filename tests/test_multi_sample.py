"""Tests for preprocessing.multi_sample (Phase 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from gtrends_bayes.preprocessing.multi_sample import average_samples


def test_average_samples_returns_wide_format(synthetic_multi_sample):
    """Output should be wide: one column per query, one row per date."""
    out = average_samples(synthetic_multi_sample, drop_high_variance=False)
    assert isinstance(out.index, pd.DatetimeIndex)
    # Both queries present when high-var filtering is off.
    assert set(out.columns) == {"clean", "noisy"}
    # Rows = unique dates.
    assert len(out) == synthetic_multi_sample["date"].nunique()


def test_average_samples_drops_noisy_query(synthetic_multi_sample):
    """The 'noisy' fixture series (sigma=30, clipped to [1,100]) has cross-sample
    std ~24; threshold=20 should drop it while keeping the clean (sigma=1) series."""
    out = average_samples(synthetic_multi_sample, drop_high_variance=True, var_threshold=20.0)
    assert "clean" in out.columns
    assert "noisy" not in out.columns


def test_average_samples_actually_averages():
    """Mean across sample_idx must equal what the function returns."""
    df = pd.DataFrame({
        "date": ["2020-01-05"] * 3 + ["2020-01-12"] * 3,
        "query": ["q"] * 6,
        "sample_idx": [0, 1, 2, 0, 1, 2],
        "svi": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    out = average_samples(df, drop_high_variance=False)
    assert out.loc[pd.Timestamp("2020-01-05"), "q"] == 20.0
    assert out.loc[pd.Timestamp("2020-01-12"), "q"] == 50.0


def test_average_samples_handles_empty_input():
    df = pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])
    out = average_samples(df)
    assert out.empty


def test_average_samples_rejects_missing_columns():
    df = pd.DataFrame({"date": [], "svi": []})  # missing query, sample_idx
    with pytest.raises(ValueError, match="missing required columns"):
        average_samples(df)
