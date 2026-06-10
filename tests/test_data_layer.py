"""Tests for the Phase 2 data layer (config loaders, cache, trends_client glue)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from gtrends_bayes.config import PredictorsConfig, TargetsConfig, ModelConfig
from gtrends_bayes.data.cache import (
    cache_path,
    read_sample,
    slugify,
    write_sample,
)
from gtrends_bayes.data import trends_client


# ---- config loaders ---------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_predictors_config_loads_and_flattens():
    cfg = PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")
    assert cfg.geo == "US"
    assert len(cfg.predictors) > 30, "expected ~25 categories + ~20 topics"
    kinds = {p.kind for p in cfg.predictors}
    assert kinds == {"category", "topic"}
    # Every category entry has int id, every topic has mid string.
    for p in cfg.predictors:
        if p.kind == "category":
            assert isinstance(p.id, int)
            assert p.mid is None
        else:
            # Topic mids may begin with /m/ (legacy Knowledge Graph) or /g/
            # (newer Google identifier — e.g. /g/1211cg58 for Economic crisis).
            assert p.mid and (p.mid.startswith("/m/") or p.mid.startswith("/g/"))
            assert p.id is None


def test_targets_config_loads():
    cfg = TargetsConfig.from_yaml(CONFIG_DIR / "targets.yaml")
    names = {t.name for t in cfg.targets}
    assert {"HY", "IG"}.issubset(names)
    # ETF proxies (HYG/LQD) replaced FRED OAS series (which FRED truncated to 2023+).
    assert any(t.ticker == "HYG" for t in cfg.targets)
    assert any(t.ticker == "LQD" for t in cfg.targets)
    assert any(c.name == "vix" for c in cfg.controls)


def test_model_config_loads():
    cfg = ModelConfig.from_yaml(CONFIG_DIR / "model.yaml")
    assert cfg.bsts.mcmc.niter == 3000
    assert cfg.backtest.refit_every == 13


# ---- cache helpers ----------------------------------------------------------

def test_slugify_string_keyword():
    assert slugify("Job Listings") == "job_listings"


def test_slugify_topic_mid():
    assert slugify("/m/01jwbf") == "m_01jwbf"


def test_slugify_category_id():
    assert slugify(60) == "cat_60"


def test_cache_path_structure(tmp_path: Path):
    p = cache_path("cat_60", "US", date(2020, 1, 1), date(2024, 12, 31), 3, root=tmp_path)
    assert p.parts[-3:] == ("US", "2020-01-01_2024-12-31", "sample_03.parquet")


def test_write_then_read_sample(tmp_path: Path):
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="W-SUN"),
        "query": ["foo"] * 5,
        "sample_idx": [0] * 5,
        "svi": [10, 20, 30, 40, 50],
    })
    p = tmp_path / "out.parquet"
    write_sample(df, p)
    round_trip = read_sample(p)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), round_trip.reset_index(drop=True))


# ---- pull_series with mocked pytrends ---------------------------------------

def _fake_iot_response() -> pd.DataFrame:
    """Mimic pytrends.interest_over_time() return shape."""
    idx = pd.date_range("2020-01-05", periods=8, freq="W-SUN", name="date")
    return pd.DataFrame({"60": [50, 52, 49, 55, 60, 58, 56, 54], "isPartial": [False] * 8}, index=idx)


def _make_mock_client(response: pd.DataFrame) -> MagicMock:
    client = MagicMock()
    client.build_payload = MagicMock()
    client.interest_over_time = MagicMock(return_value=response)
    return client


def test_pull_series_caches_and_skips_on_second_call(tmp_path: Path, monkeypatch):
    client = _make_mock_client(_fake_iot_response())
    df1 = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2020, 1, 1), end=date(2020, 3, 1),
        n_samples=2, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert client.interest_over_time.call_count == 2
    assert {"date", "query", "sample_idx", "svi"} == set(df1.columns)
    assert set(df1["sample_idx"].unique()) == {0, 1}

    # Second call should hit the cache for both samples — no new API calls.
    client.interest_over_time.reset_mock()
    df2 = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2020, 1, 1), end=date(2020, 3, 1),
        n_samples=2, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert client.interest_over_time.call_count == 0
    pd.testing.assert_frame_equal(
        df1.sort_values(["sample_idx", "date"]).reset_index(drop=True),
        df2.sort_values(["sample_idx", "date"]).reset_index(drop=True),
    )


def test_pull_series_swallows_failure_and_continues(tmp_path: Path):
    """A failing API call on sample 0 should not abort the run; sample 1 still fetches."""
    client = MagicMock()
    client.build_payload = MagicMock()
    # NB: avoid words "rate"/"limit" so the 429-retry heuristic isn't triggered.
    client.interest_over_time = MagicMock(
        side_effect=[RuntimeError("network unreachable"), _fake_iot_response()]
    )
    df = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2020, 1, 1), end=date(2020, 3, 1),
        n_samples=2, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert set(df["sample_idx"].unique()) == {1}, "only the surviving sample should appear"


def test_pull_series_returns_empty_frame_when_all_fail(tmp_path: Path):
    client = MagicMock()
    client.build_payload = MagicMock()
    client.interest_over_time = MagicMock(side_effect=RuntimeError("boom"))
    df = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2020, 1, 1), end=date(2020, 3, 1),
        n_samples=2, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert df.empty
    assert list(df.columns) == ["date", "query", "sample_idx", "svi"]
