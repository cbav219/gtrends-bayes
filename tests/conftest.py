"""Shared pytest fixtures.

Synthetic Trends-like series with known statistical properties (trend,
seasonality, structural break) so that each preprocessing function can be
unit-tested against a property it should remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DEFAULT_SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture(scope="session")
def weekly_index() -> pd.DatetimeIndex:
    """Sunday-anchored weekly index spanning 2008-01-06 .. 2026-04-26."""
    return pd.date_range("2008-01-06", "2026-04-26", freq="W-SUN")


@pytest.fixture
def synthetic_svi(weekly_index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """SVI-like series with: linear downward drift + 52-week seasonality + noise.

    Properties the preprocessing pipeline should successfully remove:
      - Linear drift (handled by bias_removal.remove_long_term_drift)
      - Annual seasonality (handled by seasonality.yoy_log_diff)
    """
    n = len(weekly_index)
    t = np.arange(n)
    drift = -0.02 * t                              # downward drift over time
    seasonal = 8.0 * np.sin(2 * np.pi * t / 52)    # annual cycle
    base = 60.0
    noise = rng.normal(0, 1.5, size=n)
    raw = base + drift + seasonal + noise
    raw = np.clip(raw, 1.0, 100.0)                 # keep within SVI range
    return pd.Series(raw, index=weekly_index, name="synthetic")


@pytest.fixture
def synthetic_multi_sample(
    weekly_index: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Multi-sample long-form draws of two synthetic queries (one noisy, one clean)."""
    n = len(weekly_index)
    rows = []
    for query, sigma in [("clean", 1.0), ("noisy", 30.0)]:
        for sample_idx in range(6):
            svi = np.clip(
                50.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 52)
                + rng.normal(0, sigma, size=n),
                1.0,
                100.0,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "date": weekly_index,
                        "query": query,
                        "sample_idx": sample_idx,
                        "svi": svi,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def synthetic_with_break(
    weekly_index: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.Series:
    """Series with an artificial level jump at 2011-01-02 (the OECD break date)."""
    n = len(weekly_index)
    base = 50.0 + rng.normal(0, 1.0, size=n)
    series = pd.Series(base, index=weekly_index, name="break_test")
    series.loc[series.index >= "2011-01-01"] += 10.0   # injected break
    return series
