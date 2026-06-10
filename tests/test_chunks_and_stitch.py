"""Tests for chunked weekly pulls + stitching (Phase 2.5)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from gtrends_bayes.data import trends_client
from gtrends_bayes.data.chunks import make_chunks
from gtrends_bayes.data.stitch import stitch_chunks


# ---- chunks.make_chunks -----------------------------------------------------

def test_make_chunks_short_window_returns_single_chunk():
    chunks = make_chunks(date(2022, 1, 1), date(2024, 6, 30))
    assert chunks == [(date(2022, 1, 1), date(2024, 6, 30))]


def test_make_chunks_full_locked_window():
    """The 2008-01-01..2026-04-30 window should split into multiple ≤54-month chunks."""
    chunks = make_chunks(date(2008, 1, 1), date(2026, 4, 30))
    assert len(chunks) >= 4
    # Every chunk must be ≤ 54 months long (the default chunk size).
    for cs, ce in chunks:
        days = (ce - cs).days
        assert days <= 54 * 31, f"chunk ({cs}..{ce}) is {days} days, > 54 months"
    # The last chunk's end must equal the requested end.
    assert chunks[-1][1] == date(2026, 4, 30)
    # Consecutive chunks must overlap (next.start < prev.end).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[0] < prev[1], f"chunks {prev} and {nxt} do not overlap"


def test_make_chunks_rejects_bad_inputs():
    import pytest

    with pytest.raises(ValueError):
        make_chunks(date(2024, 1, 1), date(2024, 1, 1))   # end == start
    with pytest.raises(ValueError):
        make_chunks(date(2024, 1, 1), date(2024, 6, 1), chunk_months=12, overlap_months=12)


# ---- stitch.stitch_chunks ---------------------------------------------------

def _mk_chunk(start: str, end: str, value: float, sample_idx: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="W-SUN")
    return pd.DataFrame({
        "date": idx,
        "query": "60",
        "sample_idx": sample_idx,
        "svi": np.full(len(idx), value, dtype=float),
    })


def test_stitch_chunks_passthrough_for_single_chunk():
    chunk = _mk_chunk("2020-01-05", "2020-12-27", value=80.0)
    out = stitch_chunks([chunk])
    # rescale_to_100 is True by default; constant 80 should rescale to 100.
    assert np.isclose(out["svi"].max(), 100.0)
    assert len(out) == len(chunk)


def test_stitch_chunks_rescales_overlap_to_match():
    """Chunk B with half the magnitude of A should be scaled up by 2 in the overlap."""
    a = _mk_chunk("2020-01-05", "2021-06-27", value=50.0)
    b = _mk_chunk("2021-01-03", "2022-12-25", value=25.0)  # 6-month overlap with a
    out = stitch_chunks([a, b], rescale_to_100=False)
    # After stitching, b's values should match a's (50) in the overlap and beyond.
    assert np.isclose(out["svi"].max(), 50.0, atol=1e-6)
    assert np.isclose(out["svi"].min(), 50.0, atol=1e-6)
    assert len(out) > len(a) and len(out) > len(b)
    # Date span covers the union.
    assert out["date"].min() == a["date"].min()
    assert out["date"].max() == b["date"].max()


def test_stitch_chunks_rescales_final_max_to_100():
    a = _mk_chunk("2020-01-05", "2021-06-27", value=50.0)
    b = _mk_chunk("2021-01-03", "2022-12-25", value=25.0)
    out = stitch_chunks([a, b], rescale_to_100=True)
    assert np.isclose(out["svi"].max(), 100.0, atol=1e-6)


def test_stitch_chunks_handles_empty_list():
    out = stitch_chunks([])
    assert out.empty
    assert list(out.columns) == ["date", "query", "sample_idx", "svi"]


def test_stitch_chunks_skips_empty_chunks():
    a = _mk_chunk("2020-01-05", "2020-06-28", value=70.0)
    empty = pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])
    out = stitch_chunks([empty, a, empty])
    assert len(out) == len(a)


# ---- trends_client.pull_series chunked path ---------------------------------

def _fake_iot_for_window(start: date, end: date, base_value: int) -> pd.DataFrame:
    """Mimic pytrends.interest_over_time() for a chunk."""
    idx = pd.date_range(start, end, freq="W-SUN", name="date")
    # Use a slowly-varying values so the stitcher has something non-trivial to align.
    n = len(idx)
    values = np.linspace(base_value, base_value + 10, n).astype(float)
    return pd.DataFrame({"60": values, "isPartial": [False] * n}, index=idx)


def test_pull_series_chunks_long_window_uses_multiple_api_calls(tmp_path: Path):
    """An 18-year window should issue multiple chunk pulls per sample."""
    client = MagicMock()
    client.build_payload = MagicMock()
    # Return a different response each time (different base value per chunk).
    responses = [_fake_iot_for_window(date(2008, 1, 1), date(2026, 4, 30), bv)
                 for bv in (50, 60, 55, 70, 65)]
    client.interest_over_time = MagicMock(side_effect=responses * 4)  # ample spares

    df = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2008, 1, 1), end=date(2026, 4, 30),
        n_samples=1, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )

    # Number of chunks for the locked window — same plan make_chunks would produce.
    expected_chunks = len(make_chunks(date(2008, 1, 1), date(2026, 4, 30)))
    assert client.interest_over_time.call_count == expected_chunks
    assert {"date", "query", "sample_idx", "svi"} == set(df.columns)
    # Stitched output should span the full window.
    assert df["date"].min().date() <= date(2008, 1, 31)
    assert df["date"].max().date() >= date(2026, 3, 1)


def test_pull_series_retries_on_rate_limit(tmp_path: Path, monkeypatch):
    """A 429 from pytrends should trigger backoff + retry without losing the chunk."""
    monkeypatch.setattr(trends_client.time, "sleep", lambda _s: None)
    monkeypatch.setattr(trends_client, "_DEFAULT_BACKOFF_SECONDS", (1, 1, 1))
    client = MagicMock()
    client.build_payload = MagicMock()
    rate_limit_exc = Exception("The request failed: Google returned a response with code 429")
    success_response = _fake_iot_for_window(date(2022, 1, 1), date(2024, 6, 30), 50)
    # Window is short enough to be a single chunk → 1 effective fetch with retry.
    client.interest_over_time = MagicMock(side_effect=[rate_limit_exc, success_response])

    df = trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2022, 1, 1), end=date(2024, 6, 30),
        n_samples=1, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert client.interest_over_time.call_count == 2  # one 429, one success
    assert not df.empty


def test_pull_series_chunked_uses_chunk_cache_on_second_run(tmp_path: Path):
    client = MagicMock()
    client.build_payload = MagicMock()
    responses = [_fake_iot_for_window(date(2008, 1, 1), date(2026, 4, 30), bv)
                 for bv in (50, 60, 55, 70, 65)]
    client.interest_over_time = MagicMock(side_effect=responses * 4)

    trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2008, 1, 1), end=date(2026, 4, 30),
        n_samples=1, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    first_calls = client.interest_over_time.call_count

    # Second run: stitched cache should hit; no new API calls.
    client.interest_over_time.reset_mock()
    trends_client.pull_series(
        query=60, kind="category", geo="US",
        start=date(2008, 1, 1), end=date(2026, 4, 30),
        n_samples=1, sleep_seconds=0,
        cache_root=tmp_path, pytrends_client=client,
    )
    assert client.interest_over_time.call_count == 0
    assert first_calls > 0
