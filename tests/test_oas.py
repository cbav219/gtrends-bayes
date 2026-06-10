"""Unit tests for data/oas.py — fully mocked, no live WRDS / FRED calls."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.data import oas as oas_mod


def _make_wrds_probe(date_col: str = "date", val_col: str = "oas",
                     idx_col: str = "index_code") -> pd.DataFrame:
    return pd.DataFrame(
        {date_col: [pd.Timestamp("2008-01-02")], val_col: [400.0], idx_col: ["H0A0"]}
    )


def _make_wrds_full(n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2008-01-02", periods=n)
    return pd.DataFrame(
        {"date": idx.astype(str), "oas_bps": np.linspace(300, 500, n)}
    )


def _patch_wrds(monkeypatch, probe_df, full_df, lib="ice_baml", table="baml_oas_indices"):
    """Install a fake wrds.Connection that returns the given probe + pull."""
    fake_conn = MagicMock()
    fake_conn.list_tables.return_value = [table, "other_table"]

    def raw_sql(sql, **kwargs):
        if "LIMIT 1" in sql:
            return probe_df
        return full_df

    fake_conn.raw_sql.side_effect = raw_sql
    fake_conn.close = MagicMock()

    def fake_connect():
        return fake_conn

    monkeypatch.setattr(oas_mod, "_wrds_connect", fake_connect)
    return fake_conn


def test_fetch_oas_wrds_happy_path(monkeypatch, tmp_path):
    probe = _make_wrds_probe()
    full = _make_wrds_full(n=20)
    _patch_wrds(monkeypatch, probe, full)

    s = oas_mod.fetch_oas(
        "HY", date(2008, 1, 1), date(2008, 2, 1),
        source="wrds", cache_root=tmp_path,
    )
    assert s.name == "HY_OAS"
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == 20
    assert s.iloc[0] == pytest.approx(300.0)
    # Cache file should now exist.
    cache_files = list(tmp_path.glob("HY_OAS_wrds_*.parquet"))
    assert len(cache_files) == 1


def test_fetch_oas_wrds_uses_cache_on_second_call(monkeypatch, tmp_path):
    probe = _make_wrds_probe()
    full = _make_wrds_full(n=10)
    fake_conn = _patch_wrds(monkeypatch, probe, full)

    oas_mod.fetch_oas("HY", date(2008, 1, 1), date(2008, 2, 1),
                      source="wrds", cache_root=tmp_path)
    first_call_count = fake_conn.raw_sql.call_count

    # Second call should hit the cache; no further raw_sql calls.
    s = oas_mod.fetch_oas("HY", date(2008, 1, 1), date(2008, 2, 1),
                          source="wrds", cache_root=tmp_path)
    assert fake_conn.raw_sql.call_count == first_call_count
    assert len(s) == 10


def test_fetch_oas_missing_wrds_username(monkeypatch, tmp_path):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        oas_mod.fetch_oas("HY", date(2008, 1, 1), date(2008, 2, 1),
                          source="wrds", cache_root=tmp_path, use_cache=False)


def test_fetch_oas_invalid_source(tmp_path):
    with pytest.raises(ValueError, match="source"):
        oas_mod.fetch_oas("HY", date(2008, 1, 1), date(2008, 2, 1),
                          source="bogus", cache_root=tmp_path)  # type: ignore[arg-type]


def test_stitch_oas_aligns_and_raises_on_divergence():
    idx_overlap = pd.bdate_range("2023-05-02", "2023-12-31")
    wrds = pd.Series(
        np.full(len(idx_overlap) + 30, 400.0),
        index=pd.bdate_range("2023-04-01", periods=len(idx_overlap) + 30),
        name="HY_OAS",
    )
    # FRED reports 410 bps (10 bps higher) on the same dates → > 5 bps gap → raise.
    fred = pd.Series(
        np.full(len(idx_overlap) + 40, 410.0),
        index=pd.bdate_range("2023-05-02", periods=len(idx_overlap) + 40),
        name="HY_OAS",
    )
    with pytest.raises(RuntimeError, match="diverged"):
        oas_mod._stitch_oas(wrds, fred)


def test_stitch_oas_succeeds_within_tolerance():
    wrds = pd.Series(
        np.full(200, 400.0),
        index=pd.bdate_range("2023-04-01", periods=200),
        name="HY_OAS",
    )
    # FRED reports 402 bps (2 bps gap, under 5 bps threshold).
    fred = pd.Series(
        np.full(220, 402.0),
        index=pd.bdate_range("2023-05-02", periods=220),
        name="HY_OAS",
    )
    out = oas_mod._stitch_oas(wrds, fred)
    # Result is WRDS for its full range, then FRED tail after WRDS ends.
    assert out.index.min() == wrds.index.min()
    assert out.index.max() == fred.index.max()
    # The overlap region should report WRDS values (400.0), not FRED (402.0).
    overlap = wrds.index.intersection(fred.index)
    assert (out.loc[overlap] == 400.0).all()


def test_resolve_wrds_table_picks_first_match(monkeypatch):
    fake_conn = MagicMock()
    # Mimic real WRDS: first candidate library raises, second succeeds.
    def list_tables(library):
        if library == "ice_baml":
            raise RuntimeError("no such library")
        if library == "bofaml":
            return ["foo_returns", "baml_oas_indices", "bar"]
        return []
    fake_conn.list_tables.side_effect = list_tables
    lib, tbl = oas_mod._resolve_wrds_table(fake_conn)
    assert lib == "bofaml"
    assert tbl == "baml_oas_indices"


def test_resolve_wrds_table_raises_when_none_match(monkeypatch):
    fake_conn = MagicMock()
    fake_conn.list_tables.return_value = ["unrelated_table"]
    with pytest.raises(RuntimeError, match="No WRDS library"):
        oas_mod._resolve_wrds_table(fake_conn)


@pytest.fixture(autouse=True)
def _ensure_wrds_username(monkeypatch):
    """Provide a dummy WRDS_USERNAME for tests that don't explicitly remove it."""
    monkeypatch.setenv("WRDS_USERNAME", "test_user")
    monkeypatch.setenv("WRDS_PASSWORD", "test_pw")
