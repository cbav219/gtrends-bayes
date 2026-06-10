"""Tests for features.library (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.config import TargetsConfig
from gtrends_bayes.features.library import (
    DEFAULT_BREAK_YEARS,
    _formula_dependencies,
    _safe_eval_formula,
    add_market_controls,
    apply_transform,
    build_feature_matrix,
    drop_low_quality_columns,
    load_market_controls,
    load_target,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# The target/control loaders read cached parquets from data/raw/targets/. That data
# is proprietary and not shipped (see data/README.md), so these tests skip when the
# cache is absent (e.g. in CI) — mirroring the importorskip("rpy2") gate the R tests use.
TARGETS_DIR = CONFIG_DIR.parent / "data" / "raw" / "targets"
requires_target_cache = pytest.mark.skipif(
    not TARGETS_DIR.exists() or not any(TARGETS_DIR.glob("*.parquet")),
    reason="cached target parquets (data/raw/targets/) not present — data not shipped",
)


# ---- apply_transform --------------------------------------------------------

def test_apply_transform_levels_passthrough():
    s = pd.Series([1.0, 2.0, 3.0])
    pd.testing.assert_series_equal(apply_transform(s, "levels"), s)


def test_apply_transform_diff_first_row_nan():
    s = pd.Series([10.0, 12.0, 15.0])
    out = apply_transform(s, "diff")
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 2.0 and out.iloc[2] == 3.0


def test_apply_transform_log_diff():
    s = pd.Series([np.e, np.e**2, np.e**3])
    out = apply_transform(s, "log_diff")
    assert pd.isna(out.iloc[0])
    assert np.isclose(out.iloc[1], 1.0)
    assert np.isclose(out.iloc[2], 1.0)


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform(pd.Series([1.0]), "ewma")  # type: ignore[arg-type]


# ---- formula evaluation ----------------------------------------------------

def test_formula_dependencies_extracts_uppercase_tokens():
    assert _formula_dependencies("DGS10 - DGS2") == ["DGS10", "DGS2"]


def test_safe_eval_formula_subtracts_two_series():
    a = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 2])
    b = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    out = _safe_eval_formula("DGS10 - DGS2", {"DGS10": a, "DGS2": b})
    pd.testing.assert_series_equal(out, pd.Series([9.0, 18.0, 27.0], index=[0, 1, 2]))


def test_safe_eval_formula_rejects_attribute_access():
    with pytest.raises(ValueError, match="unsupported AST"):
        _safe_eval_formula("DGS10.values", {"DGS10": pd.Series([1.0])})


# ---- target / control loaders (real cached parquets) ---------------------

@pytest.fixture(scope="module")
def targets_cfg():
    return TargetsConfig.from_yaml(CONFIG_DIR / "targets.yaml")


@requires_target_cache
def test_load_target_HY_returns_series(targets_cfg):
    s = load_target("HY", targets_cfg)
    assert isinstance(s, pd.Series)
    assert s.name == "HY"
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) > 800   # ~957 weekly bars from pull_targets


def test_load_target_unknown_raises(targets_cfg):
    with pytest.raises(KeyError):
        load_target("XYZ", targets_cfg)


@requires_target_cache
def test_load_market_controls_loads_and_transforms(targets_cfg):
    controls = load_market_controls(targets_cfg)
    assert set(controls) == {"vix", "ust10y", "ust2y10y_slope"}
    # vix is log_diff: first row should be NaN.
    assert pd.isna(controls["vix"].iloc[0])
    # ust10y is diff: first row should be NaN.
    assert pd.isna(controls["ust10y"].iloc[0])
    # ust2y10y_slope is levels: first row should be finite (and roughly small in bps).
    assert np.isfinite(controls["ust2y10y_slope"].iloc[0])


@requires_target_cache
def test_load_market_controls_derived_slope_matches_manual_calc(targets_cfg):
    """Slope = DGS10 − DGS2 should match a manual computation."""
    controls = load_market_controls(targets_cfg)
    ust10y_path = CONFIG_DIR.parent / "data/raw/targets/ust10y.parquet"
    dgs2_path = CONFIG_DIR.parent / "data/raw/targets/dgs2.parquet"
    ust10y = pd.read_parquet(ust10y_path).iloc[:, 0]
    dgs2 = pd.read_parquet(dgs2_path).iloc[:, 0]
    expected = (ust10y - dgs2).dropna()
    actual = controls["ust2y10y_slope"].dropna()
    common = expected.index.intersection(actual.index)
    assert len(common) > 100
    assert np.allclose(actual.loc[common].values, expected.loc[common].values)


# ---- build_feature_matrix -------------------------------------------------

def _mk_processed(weekly_index, n_cols=3, with_front_nan=52):
    """Wide, date-indexed processed-Trends-like frame with NaN run at the front."""
    n = len(weekly_index)
    data = {}
    for i in range(n_cols):
        col = np.linspace(-0.5, 0.5, n) + np.sin(np.arange(n) / 8) * 0.1
        col[:with_front_nan] = np.nan
        data[f"q{i}"] = col
    return pd.DataFrame(data, index=weekly_index)


def test_build_feature_matrix_aligns_x_and_y(weekly_index):
    Xp = _mk_processed(weekly_index)
    y = pd.Series(np.linspace(80, 100, len(weekly_index)),
                  index=weekly_index, name="HY")
    X, y_out = build_feature_matrix(Xp, y)
    assert set(X.columns) == {"q0", "q1", "q2"}
    assert X.index.equals(y_out.index)
    # First 52 rows of X have NaN — must be dropped.
    assert len(X) <= len(weekly_index) - 52


def test_build_feature_matrix_drops_break_years(weekly_index):
    Xp = _mk_processed(weekly_index)
    y = pd.Series(1.0, index=weekly_index, name="y")
    X, _ = build_feature_matrix(Xp, y)
    excluded_years = set(DEFAULT_BREAK_YEARS)
    assert excluded_years.isdisjoint(set(X.index.year.unique()))


def test_build_feature_matrix_uses_train_eligible_when_provided(weekly_index):
    Xp = _mk_processed(weekly_index)
    y = pd.Series(1.0, index=weekly_index, name="y")
    # Custom mask: drop 2020 instead of the default 2011/2016.
    mask = pd.Series(weekly_index.year != 2020, index=weekly_index)
    X, _ = build_feature_matrix(Xp, y, train_eligible=mask)
    assert 2020 not in X.index.year.unique()
    # 2011 and 2016 should now be present.
    assert 2011 in X.index.year.unique()
    assert 2016 in X.index.year.unique()


def test_build_feature_matrix_returns_empty_on_no_overlap():
    Xp = pd.DataFrame({"q0": [1.0, 2.0]},
                      index=pd.date_range("2020-01-05", periods=2, freq="W-SUN"))
    y = pd.Series([10.0, 20.0],
                  index=pd.date_range("2024-01-07", periods=2, freq="W-SUN"), name="y")
    X, y_out = build_feature_matrix(Xp, y)
    assert X.empty and y_out.empty


# ---- drop_low_quality_columns ---------------------------------------------

def test_drop_low_quality_columns_threshold():
    idx = pd.date_range("2020-01-05", periods=10, freq="W-SUN")
    df = pd.DataFrame({
        "good": np.linspace(0, 1, 10),
        "bad": [np.nan] * 7 + [1.0, 2.0, 3.0],   # 70% NaN
    }, index=idx)
    pruned = drop_low_quality_columns(df, nan_threshold=0.5)
    assert list(pruned.columns) == ["good"]


# ---- add_market_controls ---------------------------------------------------

def test_add_market_controls_appends_columns_and_returns_names(weekly_index):
    Xp = _mk_processed(weekly_index, n_cols=2, with_front_nan=0)
    controls = {
        "vix": pd.Series(np.linspace(0.01, 0.02, len(weekly_index)), index=weekly_index),
        "ust10y": pd.Series(np.linspace(0.001, -0.001, len(weekly_index)), index=weekly_index),
    }
    augmented, names = add_market_controls(Xp, controls)
    assert names == ["vix", "ust10y"]
    assert set(augmented.columns) == {"q0", "q1", "vix", "ust10y"}
    assert len(augmented) == len(Xp)


def test_add_market_controls_empty_controls_passthrough(weekly_index):
    Xp = _mk_processed(weekly_index, n_cols=1, with_front_nan=0)
    out, names = add_market_controls(Xp, {})
    assert names == []
    pd.testing.assert_frame_equal(out, Xp)
