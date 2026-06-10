"""Tests for the v2 metrics added in Phase A (IMPLEMENTATION_PLAN_v2.md §2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.backtest.metrics import (
    auc_roc,
    brier_score,
    conditional_hit_rate,
    coverage_calibration,
    information_coefficient,
    precision_recall_widening,
)


# ---- brier_score -----------------------------------------------------------

def test_brier_score_perfect_prediction_is_zero():
    y = pd.Series([0, 1, 0, 1, 1])
    p = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0])
    assert brier_score(p, y) == pytest.approx(0.0)


def test_brier_score_uninformative_is_quarter():
    y = pd.Series([0, 1, 0, 1])
    p = pd.Series([0.5, 0.5, 0.5, 0.5])
    assert brier_score(p, y) == pytest.approx(0.25)


def test_brier_score_anti_correct_is_one():
    y = pd.Series([0, 1, 0, 1])
    p = pd.Series([1.0, 0.0, 1.0, 0.0])
    assert brier_score(p, y) == pytest.approx(1.0)


def test_brier_score_handles_index_mismatch():
    y = pd.Series([0, 1, 0], index=[0, 1, 2])
    p = pd.Series([0.0, 1.0], index=[0, 1])
    assert brier_score(p, y) == pytest.approx(0.0)


# ---- auc_roc ---------------------------------------------------------------

def test_auc_roc_perfect_separator_is_one():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    p = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc_roc(p, y) == pytest.approx(1.0)


def test_auc_roc_random_is_half():
    rng = np.random.default_rng(0)
    n = 10000
    y = pd.Series(rng.integers(0, 2, n))
    p = pd.Series(rng.uniform(0, 1, n))
    assert abs(auc_roc(p, y) - 0.5) < 0.05


def test_auc_roc_single_class_returns_nan():
    y = pd.Series([0, 0, 0, 0])
    p = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(auc_roc(p, y))


# ---- information_coefficient ----------------------------------------------

def test_ic_perfect_alignment_is_one():
    rng = np.random.default_rng(0)
    y_actual = pd.Series(np.cumsum(rng.standard_normal(200)))
    y_pred = y_actual.copy()  # diffs are identical -> rank corr = 1
    assert information_coefficient(y_pred, y_actual) == pytest.approx(1.0, abs=1e-9)


def test_ic_anti_aligned_is_minus_one():
    rng = np.random.default_rng(0)
    y_actual = pd.Series(np.cumsum(rng.standard_normal(200)))
    # Build y_pred whose diffs are exactly -1 × y_actual diffs.
    y_pred = pd.Series(-y_actual.diff().fillna(0).cumsum().values, index=y_actual.index)
    ic = information_coefficient(y_pred, y_actual)
    assert ic == pytest.approx(-1.0, abs=1e-9)


def test_ic_uncorrelated_near_zero():
    rng = np.random.default_rng(0)
    y_actual = pd.Series(rng.standard_normal(2000).cumsum())
    y_pred = pd.Series(rng.standard_normal(2000).cumsum())
    ic = information_coefficient(y_pred, y_actual)
    assert abs(ic) < 0.10


def test_ic_pearson_method_works():
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.standard_normal(100)))
    ic_s = information_coefficient(y, y, method="spearman")
    ic_p = information_coefficient(y, y, method="pearson")
    assert ic_s == pytest.approx(1.0, abs=1e-9)
    assert ic_p == pytest.approx(1.0, abs=1e-9)


def test_ic_unknown_method_raises():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="unknown method"):
        information_coefficient(y, y, method="kendall")  # type: ignore[arg-type]


# ---- conditional_hit_rate -------------------------------------------------

def test_conditional_hit_rate_perfect_when_signs_match():
    y_actual = pd.Series([10.0, 12.0, 9.0, 14.0, 8.0])  # diffs: +2, -3, +5, -6
    y_pred = pd.Series([10.0, 11.0, 10.0, 13.0, 11.0])   # diffs: +1, -1, +3, -2 (all match)
    out = conditional_hit_rate(y_pred, y_actual, move_threshold=0.0)
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["n_eligible"] == 4


def test_conditional_hit_rate_filters_to_threshold():
    y_actual = pd.Series([10.0, 10.5, 13.0, 12.5, 8.0])  # diffs: +0.5, +2.5, -0.5, -4.5
    y_pred = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])   # diffs: +1, +1, +1, +1
    out = conditional_hit_rate(y_pred, y_actual, move_threshold=2.0)
    # Only diffs at indices 2 (+2.5) and 4 (-4.5) qualify; pred signs are +,+ -> 1/2 hits
    assert out["n_eligible"] == 2
    assert out["hit_rate"] == pytest.approx(0.5)


def test_conditional_hit_rate_raises_when_no_eligible():
    y_actual = pd.Series([10.0, 10.1, 10.2, 10.1])
    y_pred = pd.Series([10.0, 10.5, 10.5, 9.5])
    with pytest.raises(ValueError, match="no weeks with"):
        conditional_hit_rate(y_pred, y_actual, move_threshold=10.0)


# ---- precision_recall_widening --------------------------------------------

def test_precision_recall_widening_decrease_direction():
    """ETF-price convention: 'widening' = price drops by > threshold."""
    # Δy: NaN, -50 (event), -10, -30 (event), +5
    y_actual = pd.Series([100.0, 50.0, 40.0, 10.0, 15.0])
    # Predict event at indices 1 (TP), 2 (FP), 3 (TP), 4 (FP)
    y_pred_widening = pd.Series([0, 1, 1, 1, 1])
    out = precision_recall_widening(y_pred_widening, y_actual,
                                    widening_threshold=25.0, direction="decrease")
    assert out["n_events"] == 2
    assert out["n_predicted"] == 4
    assert out["precision"] == pytest.approx(2 / 4)
    assert out["recall"] == pytest.approx(2 / 2)


def test_precision_recall_widening_increase_direction():
    """OAS convention: widening = spread INCREASES by > threshold."""
    y_actual = pd.Series([100.0, 150.0, 160.0, 130.0, 135.0])  # Δ: +50, +10, -30, +5
    y_pred_widening = pd.Series([0, 1, 0, 0, 1])
    out = precision_recall_widening(y_pred_widening, y_actual,
                                    widening_threshold=25.0, direction="increase")
    assert out["n_events"] == 1   # only +50 qualifies
    assert out["precision"] == pytest.approx(1 / 2)


def test_precision_recall_widening_no_events_returns_nan():
    y_actual = pd.Series([100.0, 100.5, 99.5, 100.2])
    y_pred_widening = pd.Series([0, 1, 0, 1])
    out = precision_recall_widening(y_pred_widening, y_actual,
                                    widening_threshold=25.0, direction="decrease")
    assert out["n_events"] == 0
    assert np.isnan(out["precision"])
    assert np.isnan(out["recall"])


def test_precision_recall_widening_unknown_direction_raises():
    y_actual = pd.Series([100.0, 50.0])
    y_pred_widening = pd.Series([0, 1])
    with pytest.raises(ValueError, match="unknown direction"):
        precision_recall_widening(y_pred_widening, y_actual, direction="lateral")  # type: ignore[arg-type]


# ---- coverage_calibration -------------------------------------------------

def test_coverage_calibration_full_coverage():
    idx = pd.RangeIndex(100)
    y = pd.Series(np.zeros(100), index=idx)
    lo = pd.Series(-10.0, index=idx)
    hi = pd.Series(10.0, index=idx)
    out = coverage_calibration(y, lo, hi, nominal_level=0.80)
    assert out["empirical"] == pytest.approx(1.0)
    assert out["gap"] == pytest.approx(0.80 - 1.0)


def test_coverage_calibration_zero_coverage():
    idx = pd.RangeIndex(100)
    y = pd.Series(np.full(100, 100.0), index=idx)
    lo = pd.Series(-10.0, index=idx)
    hi = pd.Series(10.0, index=idx)
    out = coverage_calibration(y, lo, hi, nominal_level=0.80)
    assert out["empirical"] == pytest.approx(0.0)
    assert out["gap"] == pytest.approx(0.80)


def test_coverage_calibration_partial():
    idx = pd.RangeIndex(10)
    y = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    lo = pd.Series(-1.0, index=idx)
    hi = pd.Series(1.0, index=idx)
    out = coverage_calibration(y, lo, hi, nominal_level=0.80)
    assert out["empirical"] == pytest.approx(0.5)
    assert out["gap"] == pytest.approx(0.30)


def test_coverage_calibration_handles_no_overlap():
    y = pd.Series([0.0], index=[0])
    lo = pd.Series([-1.0], index=[10])  # disjoint
    hi = pd.Series([1.0], index=[10])
    out = coverage_calibration(y, lo, hi, nominal_level=0.80)
    assert np.isnan(out["empirical"])
