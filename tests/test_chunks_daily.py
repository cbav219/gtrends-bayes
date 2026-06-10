"""Tests for the daily-cadence chunker (data/chunks_daily.py)."""

from __future__ import annotations

from datetime import date

import pytest

from gtrends_bayes.data.chunks_daily import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_OVERLAP_DAYS,
    make_chunks_daily,
)


def test_short_window_returns_single_chunk():
    chunks = make_chunks_daily(date(2024, 1, 1), date(2024, 2, 1))
    assert chunks == [(date(2024, 1, 1), date(2024, 2, 1))]


def test_chunk_window_boundaries_are_correct():
    chunks = make_chunks_daily(
        date(2024, 1, 1), date(2024, 6, 1),
        chunk_days=80, overlap_days=10,
    )
    assert chunks[0][0] == date(2024, 1, 1)
    # Final chunk ends exactly at the requested end (no overrun).
    assert chunks[-1][1] == date(2024, 6, 1)
    # Each non-final chunk is exactly chunk_days long.
    for s, e in chunks[:-1]:
        assert (e - s).days == 80
    # Consecutive chunks overlap by exactly overlap_days.
    for prev, nxt in zip(chunks[:-1], chunks[1:]):
        gap = (nxt[0] - prev[1]).days
        # nxt starts (chunk_days - overlap_days) days after prev started =>
        # nxt[0] - prev[1] = -overlap_days (overlap = 10 days *backwards*).
        assert gap == -DEFAULT_OVERLAP_DAYS


def test_full_18yr_window_chunk_count_is_reasonable():
    # 2008-01-01 → 2026-04-30 ≈ 6,690 days; with step = 70 days, ~96 chunks.
    chunks = make_chunks_daily(date(2008, 1, 1), date(2026, 4, 30))
    assert 80 <= len(chunks) <= 110


def test_rejects_invalid_window():
    with pytest.raises(ValueError, match="end"):
        make_chunks_daily(date(2024, 6, 1), date(2024, 1, 1))


def test_rejects_overlap_ge_chunk():
    with pytest.raises(ValueError, match="overlap"):
        make_chunks_daily(date(2024, 1, 1), date(2024, 6, 1),
                          chunk_days=10, overlap_days=10)


def test_rejects_chunk_above_90_days():
    # Pytrends switches to weekly bars at ~90 days; we must not exceed that.
    with pytest.raises(ValueError, match="weekly"):
        make_chunks_daily(date(2024, 1, 1), date(2024, 6, 1),
                          chunk_days=120, overlap_days=10)


def test_default_chunk_days_safely_under_pytrends_threshold():
    assert DEFAULT_CHUNK_DAYS < 90
