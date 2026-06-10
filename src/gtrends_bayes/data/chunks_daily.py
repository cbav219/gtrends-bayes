"""Plan overlapping date-window chunks for *daily*-resolution Google Trends pulls.

Pytrends returns daily bars for windows of 7–90 days (the API switches to
weekly above ~90 days). To get continuous daily SVI over the locked
2008-01 → 2026-04 window we pull in overlapping ≤90-day chunks and stitch
them by overlap-mean rescaling (see ``data/stitch.py``).

Defaults: 80-day chunks with a 10-day overlap. That gives ~94 chunks
spanning 18 years and ≥2 stable observations per chunk for the rescale.
"""

from __future__ import annotations

from datetime import date, timedelta

# Chosen to stay safely under the 90-day weekly/daily transition (API gives
# daily only for windows ≤ ~90 days) while leaving a 10-day overlap that
# yields ~7 observations for the rescale (excluding weekends).
DEFAULT_CHUNK_DAYS = 80
DEFAULT_OVERLAP_DAYS = 10


def make_chunks_daily(
    start: date,
    end: date,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> list[tuple[date, date]]:
    """Plan overlapping daily chunks covering ``[start, end]``.

    Each chunk is at most ``chunk_days`` long; consecutive chunks overlap by
    ``overlap_days``. The final chunk's end is capped at ``end``. If the full
    window is already shorter than ``chunk_days``, returns a single chunk
    equal to ``(start, end)``.

    Parameters
    ----------
    start, end : date
    chunk_days : int, default 80
    overlap_days : int, default 10

    Returns
    -------
    list of (chunk_start, chunk_end) tuples, in chronological order.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")
    if chunk_days <= overlap_days:
        raise ValueError(
            f"chunk_days ({chunk_days}) must exceed overlap_days ({overlap_days})"
        )
    if chunk_days > 90:
        raise ValueError(
            f"chunk_days ({chunk_days}) > 90 — pytrends would return weekly bars, "
            "defeating the purpose of daily chunking."
        )

    total_days = (end - start).days
    if total_days <= chunk_days:
        return [(start, end)]

    step = chunk_days - overlap_days
    chunks: list[tuple[date, date]] = []
    current_start = start
    while True:
        chunk_end = min(end, current_start + timedelta(days=chunk_days))
        chunks.append((current_start, chunk_end))
        if chunk_end >= end:
            break
        current_start = current_start + timedelta(days=step)
    return chunks
