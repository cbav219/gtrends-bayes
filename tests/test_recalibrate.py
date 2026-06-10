"""Tests for the conformal recalibration module (Phase D.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.backtest.recalibrate import (
    apply_conformal_multiplier,
    fit_per_level,
    learn_conformal_multiplier,
)


# ---- learn_conformal_multiplier --------------------------------------------

def test_alpha_one_when_band_already_meets_nominal():
    """If the band already achieves nominal coverage on the data, α should be close to 1."""
    rng = np.random.default_rng(0)
    n = 1000
    idx = pd.RangeIndex(n)
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    med = pd.Series(np.zeros(n), index=idx)
    # 80% interval of N(0, 1) is [-1.282, 1.282]
    lo = pd.Series(np.full(n, -1.282), index=idx)
    hi = pd.Series(np.full(n, 1.282), index=idx)
    alpha = learn_conformal_multiplier(y, lo, hi, nominal_level=0.80, median=med)
    assert 0.9 < alpha < 1.1


def test_alpha_greater_than_one_when_undercovering():
    """If the band is too tight, α should be > 1 to inflate it."""
    rng = np.random.default_rng(0)
    n = 1000
    idx = pd.RangeIndex(n)
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    med = pd.Series(np.zeros(n), index=idx)
    # Half-width 0.5 ⇒ ~38% coverage of N(0,1); need α≈2.5 to reach 80%.
    lo = pd.Series(np.full(n, -0.5), index=idx)
    hi = pd.Series(np.full(n, 0.5), index=idx)
    alpha = learn_conformal_multiplier(y, lo, hi, nominal_level=0.80, median=med)
    assert alpha > 2.0


def test_alpha_less_than_one_when_overcovering():
    """If the band is too wide, α should be < 1 to shrink it."""
    rng = np.random.default_rng(0)
    n = 1000
    idx = pd.RangeIndex(n)
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    med = pd.Series(np.zeros(n), index=idx)
    # Half-width 5 ⇒ ~99.99% coverage; we want 80% → α should be ~0.26.
    lo = pd.Series(np.full(n, -5.0), index=idx)
    hi = pd.Series(np.full(n, 5.0), index=idx)
    alpha = learn_conformal_multiplier(y, lo, hi, nominal_level=0.80, median=med)
    assert alpha < 0.5


def test_alpha_nan_on_no_overlap():
    y = pd.Series([0.0], index=[0])
    lo = pd.Series([-1.0], index=[10])
    hi = pd.Series([1.0], index=[10])
    alpha = learn_conformal_multiplier(y, lo, hi, nominal_level=0.8)
    assert np.isnan(alpha)


# ---- apply_conformal_multiplier --------------------------------------------

def test_apply_alpha_one_is_identity():
    med = pd.Series([0.0, 0.0])
    lo = pd.Series([-1.0, -1.0])
    hi = pd.Series([1.0, 1.0])
    lo_cal, hi_cal = apply_conformal_multiplier(med, lo, hi, alpha=1.0)
    pd.testing.assert_series_equal(lo_cal, lo, check_names=False)
    pd.testing.assert_series_equal(hi_cal, hi, check_names=False)


def test_apply_alpha_two_doubles_half_widths():
    med = pd.Series([0.0, 5.0])
    lo = pd.Series([-1.0, 4.0])  # half-widths 1, 1
    hi = pd.Series([1.0, 6.0])
    lo_cal, hi_cal = apply_conformal_multiplier(med, lo, hi, alpha=2.0)
    assert (lo_cal == pd.Series([-2.0, 3.0])).all()
    assert (hi_cal == pd.Series([2.0, 7.0])).all()


def test_apply_rejects_negative_alpha():
    med = pd.Series([0.0])
    lo = pd.Series([-1.0])
    hi = pd.Series([1.0])
    with pytest.raises(ValueError, match="non-negative"):
        apply_conformal_multiplier(med, lo, hi, alpha=-0.5)


# ---- fit_per_level ---------------------------------------------------------

def _mk_bands(n: int, half_widths: dict[str, float], seed: int = 0):
    """Synthetic bands with the full WalkForward schema, centered at 0."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    bands = pd.DataFrame(index=idx)
    bands["q500"] = 0.0
    for q in (0.025, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.975):
        col = f"q{int(round(q * 1000)):03d}"
        # Use a level-specific half-width.
        if q < 0.5:
            bands[col] = -half_widths.get(col, 1.0)
        else:
            bands[col] = half_widths.get(col, 1.0)
    return y, bands


def test_fit_per_level_in_sample_default():
    """Default mode (val_split=None) returns in-sample α and exact-nominal post coverage."""
    half_widths = {"q025": 1.96, "q975": 1.96,
                   "q100": 1.282, "q900": 1.282,
                   "q250": 0.674, "q750": 0.674}
    y, bands = _mk_bands(400, half_widths)
    out = fit_per_level(y, bands, levels=(0.50, 0.80, 0.95))
    assert set(out.keys()) == {0.50, 0.80, 0.95}
    for level, d in out.items():
        assert "alpha" in d
        assert d["alpha_oos"] is None
        # In-sample α makes post-cal coverage equal to nominal by construction
        # (modulo sample granularity ~ 1/n).
        assert abs(d["empirical_post_full"] - level) <= 1.0 / d["n_full"] + 1e-9


def test_fit_per_level_oos_schema():
    half_widths = {"q025": 1.96, "q975": 1.96,
                   "q100": 1.282, "q900": 1.282,
                   "q250": 0.674, "q750": 0.674}
    y, bands = _mk_bands(400, half_widths)
    out = fit_per_level(y, bands, levels=(0.50, 0.80, 0.95), val_split=0.5)
    for level, d in out.items():
        assert d["alpha_oos"] is not None
        assert d["n_val"] + d["n_test"] == 400
        assert d["empirical_pre_test"] is not None
        assert d["empirical_post_test"] is not None


def test_fit_per_level_undercovering_gets_alpha_above_one():
    """Too-tight bands ⇒ α > 1 after recalibration."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    bands = pd.DataFrame(index=idx)
    bands["q500"] = 0.0
    # All bands undersized by 2x: 80% band is ±0.6 instead of ±1.28
    for q, hw in [(0.025, 1.0), (0.05, 0.85), (0.10, 0.65), (0.25, 0.35),
                  (0.75, 0.35), (0.90, 0.65), (0.95, 0.85), (0.975, 1.0)]:
        col = f"q{int(round(q * 1000)):03d}"
        bands[col] = -hw if q < 0.5 else hw
    out = fit_per_level(y, bands, levels=(0.80,), val_split=0.5)
    assert out[0.80]["alpha"] > 1.2


def test_fit_per_level_recalibrated_test_coverage_within_band():
    """On stationary synthetic data, OOS post-recalibration test coverage
    should land near the nominal level (within ±10pp at n=400)."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    y = pd.Series(rng.normal(0, 1, n), index=idx)
    bands = pd.DataFrame(index=idx)
    bands["q500"] = 0.0
    for q, hw in [(0.025, 1.0), (0.05, 0.85), (0.10, 0.65), (0.25, 0.35),
                  (0.75, 0.35), (0.90, 0.65), (0.95, 0.85), (0.975, 1.0)]:
        col = f"q{int(round(q * 1000)):03d}"
        bands[col] = -hw if q < 0.5 else hw
    out = fit_per_level(y, bands, levels=(0.50, 0.80, 0.95), val_split=0.5)
    for level, d in out.items():
        assert abs(d["empirical_post_test"] - level) < 0.10, (
            f"level={level}: post_test={d['empirical_post_test']:.3f}"
        )


def test_fit_per_level_rejects_too_small_split():
    y, bands = _mk_bands(20, {})
    with pytest.raises(ValueError, match="too-small slices"):
        fit_per_level(y, bands, val_split=0.05)
