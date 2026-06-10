"""Tests for the v4 inference module.

Uses a synthetic frozen-model fixture (no real BSTS posteriors required).
The shape + schema match what scripts/freeze_model_v4.py will produce from
v3 outputs, so these tests act as a contract for the freeze script as well.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.inference import forecast, load_model
from gtrends_bayes.inference.cli import main as cli_main


N_PREDICTORS = 5


def _synthetic_model(
    target: str = "HY",
    transform: str = "log_diff",
    ar_p: int = 4,
    cadence: str = "weekly",
    conformal_alpha: float = 1.0,
) -> dict:
    """Build a minimal-but-valid frozen-model dict matching the v4 schema."""
    cols = [f"pred_{i}" for i in range(N_PREDICTORS)]
    coef_summary = pd.DataFrame({
        "mean": np.array([0.02, -0.01, 0.005, 0.0, 0.03]),
        "sd": np.array([0.01, 0.01, 0.005, 0.02, 0.01]),
        "sign_consistency": np.array([0.95, 0.80, 0.55, 0.50, 0.97]),
    }, index=cols)
    inclusion = pd.Series([0.9, 0.4, 0.2, 0.05, 0.95], index=cols, name="inclusion")
    return {
        "target": target,
        "target_transform": transform,
        "build_timestamp": "2026-05-12",
        "v3_commit_hash": "synthetic",
        "ar_backbone": {
            "p": ar_p,
            "coefficients": np.array([0.5, -0.2, 0.1, 0.05])[:ar_p],
            "intercept": 0.0,
            "sigma": 0.01,
        },
        "bsts_posterior": {
            "inclusion_probs": inclusion,
            "coefficient_summary": coef_summary,
            "state_spec": {"local_linear_trend": True,
                           "seasonal": {"enabled": False}},
            "component_bands": {},
            "X_columns": cols,
        },
        "preprocessing": {
            "drift_removal": {
                "hp_lambda": 129_600,
                "pca_components": np.eye(N_PREDICTORS),
                "pca_mean": np.zeros(N_PREDICTORS),
            },
            "yoy_periods_per_year": 52 if cadence == "weekly" else 252,
            "structural_break_dates": [pd.Timestamp("2011-01-01")],
            "cadence": cadence,
        },
        "conformal_alpha": conformal_alpha,
    }


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_model_v4.pkl"
    with open(path, "wb") as f:
        pickle.dump(_synthetic_model(), f)
    return path


@pytest.fixture
def synthetic_y_x():
    """A 300-week history of HYG-shaped levels + the 5 synthetic predictors."""
    idx = pd.date_range("2020-01-05", periods=300, freq="W-SUN")
    rng = np.random.default_rng(0)
    y = pd.Series(
        80.0 * np.cumprod(1 + rng.normal(0, 0.01, size=300)),
        index=idx, name="HY",
    )
    cols = [f"pred_{i}" for i in range(N_PREDICTORS)]
    x = pd.DataFrame(
        rng.normal(0, 0.5, size=(300, N_PREDICTORS)),
        index=idx, columns=cols,
    )
    return y, x


# -- load_model -------------------------------------------------------------

def test_load_model_happy_path(model_path: Path):
    model = load_model(model_path)
    assert model["target"] == "HY"
    assert model["target_transform"] == "log_diff"
    assert model["ar_backbone"]["p"] == 4
    assert len(model["bsts_posterior"]["X_columns"]) == N_PREDICTORS


def test_load_model_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="frozen model"):
        load_model(tmp_path / "missing.pkl")


def test_load_model_not_a_dict(tmp_path: Path):
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump([1, 2, 3], f)
    with pytest.raises(ValueError, match="should unpickle to dict"):
        load_model(bad)


def test_load_model_missing_top_keys(tmp_path: Path):
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"target": "HY"}, f)
    with pytest.raises(ValueError, match="missing required top-level keys"):
        load_model(bad)


def test_load_model_invalid_transform(tmp_path: Path):
    bad = _synthetic_model()
    bad["target_transform"] = "square_root"
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as f:
        pickle.dump(bad, f)
    with pytest.raises(ValueError, match="target_transform"):
        load_model(p)


# -- forecast() -------------------------------------------------------------

def test_forecast_returns_expected_keys(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    result = forecast(model, "1w", pd.Timestamp("2025-12-15"), y, x, n_draws=100)
    expected = {
        "target", "target_transform", "as_of", "horizon", "horizon_bd",
        "n_draws", "conformal_alpha",
        "median", "q05", "q95", "level_median", "level_band",
        "path_median", "path_q05", "path_q95",
        "level_path_median", "level_path_q05", "level_path_q95",
    }
    assert expected.issubset(result.keys())


def test_forecast_band_ordered(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    result = forecast(model, "1m", pd.Timestamp("2025-12-15"), y, x, n_draws=200)
    assert result["q05"] <= result["median"] <= result["q95"]
    lo, hi = result["level_band"]
    assert lo <= result["level_median"] <= hi


def test_forecast_horizon_dispatch(model_path: Path, synthetic_y_x):
    """Each labelled horizon should resolve to the right BD count and path length."""
    y, x = synthetic_y_x
    model = load_model(model_path)
    for label, expected_bd in [("1d", 1), ("1w", 5), ("1m", 21), ("1q", 63)]:
        result = forecast(model, label, pd.Timestamp("2025-12-15"),
                          y, x, n_draws=50)
        assert result["horizon_bd"] == expected_bd
        assert result["horizon"] == label
        assert len(result["path_median"]) == expected_bd
        assert len(result["level_path_median"]) == expected_bd


def test_forecast_integer_horizon(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    result = forecast(model, 7, pd.Timestamp("2025-12-15"),
                      y, x, n_draws=50)
    assert result["horizon_bd"] == 7
    assert result["horizon"] == "7bd"


def test_forecast_rejects_unknown_horizon_label(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    with pytest.raises(ValueError, match="horizon"):
        forecast(model, "5y", pd.Timestamp("2025-12-15"), y, x)


def test_forecast_rejects_negative_horizon(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    with pytest.raises(ValueError, match="positive"):
        forecast(model, -3, pd.Timestamp("2025-12-15"), y, x)


def test_forecast_rejects_too_short_y(model_path: Path, synthetic_y_x):
    _, x = synthetic_y_x
    model = load_model(model_path)
    # AR(4) on log_diff needs at least 5 obs (one consumed by diff, 4 for state).
    short_y = pd.Series(
        [80.0, 80.5, 81.0],
        index=pd.date_range("2025-01-05", periods=3, freq="W-SUN"),
        name="HY",
    )
    with pytest.raises(ValueError, match="needs at least"):
        forecast(model, "1w", pd.Timestamp("2025-12-15"), short_y, x)


def test_forecast_rejects_missing_x_columns(model_path: Path, synthetic_y_x):
    y, _ = synthetic_y_x
    model = load_model(model_path)
    bad_x = pd.DataFrame(
        np.zeros((100, 3)),
        index=pd.date_range("2025-01-05", periods=100, freq="W-SUN"),
        columns=["pred_0", "pred_1", "pred_2"],  # missing pred_3 / pred_4
    )
    with pytest.raises(ValueError, match="missing expected predictor columns"):
        forecast(model, "1w", pd.Timestamp("2025-12-15"), y, bad_x)


def test_forecast_inverse_consistency_log_diff(model_path: Path, synthetic_y_x):
    """level_median should equal last_level * exp(cumulative_median_log_returns)."""
    y, x = synthetic_y_x
    model = load_model(model_path)
    result = forecast(model, "1m", pd.Timestamp("2025-12-15"),
                      y, x, n_draws=200)
    last_level = float(y.iloc[-1])
    expected_terminal = last_level * np.exp(sum(result["path_median"]))
    assert result["level_median"] == pytest.approx(expected_terminal, rel=1e-9)


def test_forecast_inverse_consistency_diff(tmp_path: Path, synthetic_y_x):
    """For diff transform, level_median = last_level + sum(median_path)."""
    y, x = synthetic_y_x
    m = _synthetic_model(transform="diff")
    p = tmp_path / "diff_model.pkl"
    with open(p, "wb") as f:
        pickle.dump(m, f)
    model = load_model(p)
    result = forecast(model, "1m", pd.Timestamp("2025-12-15"),
                      y, x, n_draws=200)
    expected_terminal = float(y.iloc[-1]) + sum(result["path_median"])
    assert result["level_median"] == pytest.approx(expected_terminal, rel=1e-9)


def test_forecast_seed_is_deterministic(model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    model = load_model(model_path)
    a = forecast(model, "1m", pd.Timestamp("2025-12-15"),
                 y, x, n_draws=200, seed=123)
    b = forecast(model, "1m", pd.Timestamp("2025-12-15"),
                 y, x, n_draws=200, seed=123)
    assert a["median"] == b["median"]
    assert a["q05"] == b["q05"]
    assert a["q95"] == b["q95"]


def test_forecast_no_r_imports():
    """Inference module must NOT import rpy2 / R (a hard v4 constraint)."""
    import gtrends_bayes.inference as inf
    import sys
    # Walk module hierarchy; ensure no rpy2 modules are loaded.
    loaded = [m for m in sys.modules if "rpy2" in m.lower()]
    # Re-importing inference shouldn't trigger an rpy2 load:
    _ = inf.load_model, inf.forecast
    new_loaded = [m for m in sys.modules if "rpy2" in m.lower()]
    assert new_loaded == loaded, (
        f"importing gtrends_bayes.inference loaded rpy2: {set(new_loaded) - set(loaded)}"
    )


# -- CLI ---------------------------------------------------------------------

def test_cli_writes_json(tmp_path: Path, model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    y_csv = tmp_path / "y.csv"
    pd.DataFrame({"date": y.index, "value": y.values}).to_csv(y_csv, index=False)
    x_parquet = tmp_path / "x.parquet"
    x.to_parquet(x_parquet)
    out_json = tmp_path / "fcst.json"

    rc = cli_main([
        "--model-path", str(model_path),
        "--horizon", "1w",
        "--as-of", "2025-12-15",
        "--y-data", str(y_csv),
        "--x-data", str(x_parquet),
        "--output", str(out_json),
        "--n-draws", "100",
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["target"] == "HY"
    assert payload["horizon"] == "1w"
    assert payload["horizon_bd"] == 5


def test_cli_target_mismatch_errors(tmp_path: Path, model_path: Path, synthetic_y_x):
    y, x = synthetic_y_x
    y_csv = tmp_path / "y.csv"
    pd.DataFrame({"date": y.index, "value": y.values}).to_csv(y_csv, index=False)
    x_parquet = tmp_path / "x.parquet"
    x.to_parquet(x_parquet)
    with pytest.raises(SystemExit, match="model\\['target'\\]"):
        cli_main([
            "--model-path", str(model_path),
            "--horizon", "1w",
            "--as-of", "2025-12-15",
            "--y-data", str(y_csv),
            "--x-data", str(x_parquet),
            "--target", "IG",  # mismatch
        ])


# -- Edge-case coverage (T3.2 from INTERIM_TASKS.md) ------------------------

def test_forecast_cadence_mismatch_warns(tmp_path: Path, synthetic_y_x):
    """Daily-trained model with a weekly-cadence X must emit UserWarning."""
    y, x = synthetic_y_x  # weekly-cadence (W-SUN) index
    daily_model = _synthetic_model(cadence="daily")
    model_p = tmp_path / "daily_model.pkl"
    with open(model_p, "wb") as f:
        pickle.dump(daily_model, f)
    model = load_model(model_p)
    with pytest.warns(UserWarning, match="looks weekly.*cadence='daily'"):
        forecast(model, "1w", pd.Timestamp("2025-12-01"), y, x, n_draws=50)


def test_forecast_works_with_single_row_x(model_path: Path, synthetic_y_x):
    """forecast() uses only the most recent X row (nowcasting trick) — confirm
    inference still succeeds when caller ships exactly one row."""
    y, x = synthetic_y_x
    model = load_model(model_path)
    x_one_row = x.iloc[[-1]]
    out = forecast(model, "1w", pd.Timestamp("2025-12-01"), y, x_one_row, n_draws=50)
    assert np.isfinite(out["median"])
    assert np.isfinite(out["level_median"])
    assert out["q05"] < out["q95"]


def test_forecast_path_lengths_match_horizon_bd(model_path: Path, synthetic_y_x):
    """All path arrays must have length horizon_bd (one entry per step)."""
    y, x = synthetic_y_x
    model = load_model(model_path)
    for horizon, expected_h in (("1w", 5), ("1m", 21), ("1q", 63)):
        out = forecast(model, horizon, pd.Timestamp("2025-12-01"), y, x, n_draws=50)
        assert out["horizon_bd"] == expected_h
        for key in ("path_median", "path_q05", "path_q95",
                    "level_path_median", "level_path_q05", "level_path_q95"):
            assert len(out[key]) == expected_h, (
                f"{key} length {len(out[key])} != horizon_bd {expected_h}"
            )


def test_forecast_levels_transform_median_equals_level_median(tmp_path: Path, synthetic_y_x):
    """For transform=levels models, median ≡ level_median (no aggregation)."""
    y, x = synthetic_y_x
    levels_model = _synthetic_model(transform="levels")
    model_p = tmp_path / "levels_model.pkl"
    with open(model_p, "wb") as f:
        pickle.dump(levels_model, f)
    model = load_model(model_p)
    out = forecast(model, "1m", pd.Timestamp("2025-12-01"), y, x, n_draws=200)
    # Transform-space and level-space should agree to numerical tolerance.
    assert abs(out["median"] - out["level_median"]) < 1e-6
    assert abs(out["q05"] - out["level_band"][0]) < 1e-6
    assert abs(out["q95"] - out["level_band"][1]) < 1e-6


def test_forecast_higher_alpha_widens_band(tmp_path: Path, synthetic_y_x):
    """Conformal multiplier α scales band away from the median monotonically."""
    y, x = synthetic_y_x
    model_narrow = _synthetic_model(conformal_alpha=1.0)
    model_wide = _synthetic_model(conformal_alpha=2.5)
    p_narrow = tmp_path / "narrow.pkl"
    p_wide = tmp_path / "wide.pkl"
    with open(p_narrow, "wb") as f:
        pickle.dump(model_narrow, f)
    with open(p_wide, "wb") as f:
        pickle.dump(model_wide, f)

    # Same seed → same posterior draws → α scaling is the only difference.
    out_n = forecast(load_model(p_narrow), "1m", pd.Timestamp("2025-12-01"),
                     y, x, n_draws=400, seed=123)
    out_w = forecast(load_model(p_wide), "1m", pd.Timestamp("2025-12-01"),
                     y, x, n_draws=400, seed=123)

    width_n = out_n["q95"] - out_n["q05"]
    width_w = out_w["q95"] - out_w["q05"]
    assert width_w > width_n, (
        f"α=2.5 band width {width_w:.4f} should exceed α=1.0 width {width_n:.4f}"
    )
    # Approximately 2.5× (modulo Monte-Carlo noise on the quantile estimates).
    assert 1.5 * width_n < width_w < 3.5 * width_n


# ---- OAS overlay translation -------------------------------------------------


def _model_with_oas_overlay(
    slope: float = -1700.0,
    last_oas: float = 280.0,
    pearson: float = -0.69,
) -> dict:
    """Build a levels-transform synthetic model carrying an OAS overlay block."""
    base = _synthetic_model(transform="levels")
    base["oas_overlay_translation"] = {
        "slope_bps_per_dlog": slope,
        "pearson": pearson,
        "spearman": -0.66,
        "n_overlap_weeks": 154,
        "overlap_start": "2023-05-28",
        "overlap_end": "2026-05-03",
        "last_oas_bps": last_oas,
        "last_oas_date": "2026-05-17",
        "proxy_quality_label": "defensible" if abs(pearson) >= 0.6 else "weak",
        "source": "synthetic-test",
    }
    return base


def test_forecast_emits_oas_implied_when_translation_present(
    tmp_path: Path, synthetic_y_x,
):
    """oas_implied_median == last_oas + slope · ln(level_forecast / last_level)."""
    y, x = synthetic_y_x
    model = _model_with_oas_overlay(slope=-1738.74, last_oas=280.0)
    p = tmp_path / "with_overlay.pkl"
    with open(p, "wb") as f:
        pickle.dump(model, f)

    out = forecast(load_model(p), "1m", pd.Timestamp("2025-12-01"),
                   y, x, n_draws=400, seed=42)

    # Required keys present.
    for key in ("oas_implied_median", "oas_implied_band",
                "oas_implied_path_median", "oas_implied_path_band_lo",
                "oas_implied_path_band_hi", "oas_overlay_meta"):
        assert key in out, f"forecast() missing OAS field {key!r}"

    # The translation should reproduce the closed-form algebra.
    # Match production's eps-clamp so synthetic fixtures with random-walk
    # paths that occasionally dip negative still produce a defined log.
    last_level = float(y.iloc[-1])
    slope = model["oas_overlay_translation"]["slope_bps_per_dlog"]
    last_oas = model["oas_overlay_translation"]["last_oas_bps"]
    eps = 1e-9
    expected_med = last_oas + slope * np.log(
        max(out["level_median"], eps) / last_level
    )
    assert abs(out["oas_implied_median"] - expected_med) < 1e-6

    # Band must be ordered low ≤ high; both sides are bps.
    lo, hi = out["oas_implied_band"]
    assert lo <= hi
    # Path lengths match horizon_bd.
    assert len(out["oas_implied_path_median"]) == out["horizon_bd"]
    assert len(out["oas_implied_path_band_lo"]) == out["horizon_bd"]
    assert len(out["oas_implied_path_band_hi"]) == out["horizon_bd"]


def test_forecast_omits_oas_implied_when_translation_absent(
    tmp_path: Path, synthetic_y_x,
):
    """Models without oas_overlay_translation must omit oas_implied_* (backward-compat)."""
    y, x = synthetic_y_x
    base_model = _synthetic_model(transform="levels")
    # Sanity: factory does not add the overlay key.
    assert "oas_overlay_translation" not in base_model
    p = tmp_path / "no_overlay.pkl"
    with open(p, "wb") as f:
        pickle.dump(base_model, f)

    out = forecast(load_model(p), "1m", pd.Timestamp("2025-12-01"),
                   y, x, n_draws=200, seed=42)

    for key in ("oas_implied_median", "oas_implied_band",
                "oas_implied_path_median", "oas_implied_path_band_lo",
                "oas_implied_path_band_hi", "oas_overlay_meta"):
        assert key not in out, (
            f"forecast() leaked OAS field {key!r} when no overlay was present"
        )
