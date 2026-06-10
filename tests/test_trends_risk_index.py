"""Tests for features.trends_risk_index (Phase C.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.features.trends_risk_index import build_risk_index, crisis_windows


def _mk_posterior(coefs: dict[str, tuple[float, float]]) -> dict:
    """Build a minimal posterior pickle structure from {pred: (incl_prob, beta)} pairs."""
    rows = []
    for pred, (incl, beta) in coefs.items():
        rows.append({
            "predictor": pred, "inclusion_prob": incl,
            "mean_when_included": beta, "sd_when_included": 0.05,
            "sign_consistency": 1.0,
        })
    summary = pd.DataFrame(rows).set_index("predictor")
    return {
        "coefficient_summary": summary,
        "inclusion_probs": summary["inclusion_prob"],
    }


def _mk_X(weeks: int = 300, cols: list[str] | None = None,
           seed: int = 0) -> pd.DataFrame:
    cols = cols or ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-07", periods=weeks, freq="W-SUN")
    return pd.DataFrame(rng.standard_normal((weeks, len(cols))), index=idx, columns=cols)


def test_zero_inclusion_posterior_returns_zero_index():
    posterior = _mk_posterior({"A": (0.0, 0.5), "B": (0.0, -0.3)})
    X = _mk_X()
    out = build_risk_index(posterior, X, target_kind="spread")
    assert (out["raw_index"] == 0.0).all()


def test_single_saturated_predictor_reduces_to_beta_x():
    """With one predictor at P=1 and known β, raw_index = β · X."""
    posterior = _mk_posterior({"A": (1.0, 2.0), "B": (0.0, 0.0)})
    X = _mk_X(weeks=200, cols=["A", "B"])
    out = build_risk_index(posterior, X, target_kind="spread")
    expected = 2.0 * X["A"]
    pd.testing.assert_series_equal(
        out["raw_index"].rename(None), expected.rename(None), atol=1e-9,
    )


def test_price_kind_flips_sign():
    """target_kind='price' flips the sign vs target_kind='spread'."""
    posterior = _mk_posterior({"A": (1.0, 2.0)})
    X = _mk_X(weeks=200, cols=["A"])
    spread_idx = build_risk_index(posterior, X, target_kind="spread")["raw_index"]
    price_idx = build_risk_index(posterior, X, target_kind="price")["raw_index"]
    pd.testing.assert_series_equal(
        spread_idx.rename(None), -price_idx.rename(None), atol=1e-9,
    )


def test_rolling_zscore_normalization():
    """In a regime where raw is roughly stationary, the z-score has mean ~0 and std ~1
    on the post-burn-in window."""
    rng = np.random.default_rng(0)
    posterior = _mk_posterior({"A": (1.0, 1.0)})
    idx = pd.date_range("2010-01-03", periods=600, freq="W-SUN")
    X = pd.DataFrame({"A": rng.standard_normal(600)}, index=idx)
    out = build_risk_index(posterior, X, target_kind="spread")
    valid_zs = out["zscore_5y"].dropna()
    # After the rolling window's min_periods, the z-score should have stable
    # variance properties.
    assert abs(valid_zs.iloc[-260:].mean()) < 0.5
    assert abs(valid_zs.iloc[-260:].std() - 1.0) < 0.5


def test_tiers_match_zscore_thresholds():
    posterior = _mk_posterior({"A": (1.0, 1.0)})
    X = _mk_X(weeks=300, cols=["A"])
    out = build_risk_index(posterior, X, target_kind="spread",
                            tier_thresholds=(-1.0, 1.0))
    valid = out.dropna(subset=["zscore_5y"])
    high = valid[valid["zscore_5y"] >= 1.0]
    low = valid[valid["zscore_5y"] <= -1.0]
    med = valid[(valid["zscore_5y"] > -1.0) & (valid["zscore_5y"] < 1.0)]
    assert (high["tier"] == "high").all()
    assert (low["tier"] == "low").all()
    assert (med["tier"] == "med").all()


def test_no_overlap_raises():
    posterior = _mk_posterior({"A": (1.0, 1.0)})
    X = _mk_X(cols=["X", "Y"])  # no overlap
    with pytest.raises(ValueError, match="no overlap"):
        build_risk_index(posterior, X)


def test_unknown_target_kind_raises():
    posterior = _mk_posterior({"A": (1.0, 1.0)})
    X = _mk_X(cols=["A"])
    with pytest.raises(ValueError, match="unknown target_kind"):
        build_risk_index(posterior, X, target_kind="other")  # type: ignore[arg-type]


def test_top_predictor_identified_per_week():
    """The contribution_top_predictor column should always be one of the input
    predictors, and contribution_top_value should equal contributions[t, top_pred]."""
    posterior = _mk_posterior({"A": (1.0, 2.0), "B": (1.0, 1.0)})
    X = pd.DataFrame({
        "A": [1.0, 0.1, 0.5],
        "B": [0.1, 5.0, 0.4],
    }, index=pd.date_range("2020-01-05", periods=3, freq="W-SUN"))
    out = build_risk_index(posterior, X, target_kind="spread")
    # Week 0: A·2=2, B·1=0.1 -> A wins; week 1: A·2=0.2, B·1=5 -> B wins; week 2: A wins.
    assert out["contribution_top_predictor"].iloc[0] == "A"
    assert out["contribution_top_predictor"].iloc[1] == "B"
    assert out["contribution_top_value"].iloc[0] == pytest.approx(2.0)
    assert out["contribution_top_value"].iloc[1] == pytest.approx(5.0)


def test_crisis_windows_returns_three_anchors():
    cw = crisis_windows()
    assert set(cw.keys()) >= {"COVID 2020-03", "UK gilt 2022-09", "SVB 2023-03"}
    for ts in cw.values():
        assert isinstance(ts, pd.Timestamp)
