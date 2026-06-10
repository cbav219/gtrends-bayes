"""Plan overlapping date-window chunks for Google Trends pulls.

Pytrends returns:
- weekly bars for windows of ~7 days .. ~5 years
- monthly bars for windows beyond ~5 years

To get weekly resolution over a ~18-year locked window we pull in overlapping
chunks of ≤5 years and stitch them together (see data/stitch.py).
"""

from __future__ import annotations

from datetime import date

from dateutil.relativedelta import relativedelta

# Defaults chosen to (a) stay safely under the 5-year monthly-resolution
# threshold and (b) leave a meaningful overlap for stable rescaling.
DEFAULT_CHUNK_MONTHS = 54   # 4.5 years
DEFAULT_OVERLAP_MONTHS = 12  # 1 year


def make_chunks(
    start: date,
    end: date,
    chunk_months: int = DEFAULT_CHUNK_MONTHS,
    overlap_months: int = DEFAULT_OVERLAP_MONTHS,
) -> list[tuple[date, date]]:
    """Plan overlapping chunks covering ``[start, end]``.

    Each chunk is at most ``chunk_months`` long; consecutive chunks overlap by
    ``overlap_months``. The final chunk's end is capped at ``end``. If the
    full window is already shorter than ``chunk_months``, returns a single
    chunk equal to ``(start, end)``.

    Parameters
    ----------
    start, end : date
    chunk_months : int, default 54 (4.5 years)
    overlap_months : int, default 12 (1 year)

    Returns
    -------
    list of (chunk_start, chunk_end) tuples, in chronological order.
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be strictly after start ({start})")
    if chunk_months <= overlap_months:
        raise ValueError(
            f"chunk_months ({chunk_months}) must exceed overlap_months ({overlap_months})"
        )

    full_length_months = relativedelta(end, start).years * 12 + relativedelta(end, start).months
    if full_length_months <= chunk_months:
        return [(start, end)]

    step_months = chunk_months - overlap_months
    chunks: list[tuple[date, date]] = []
    current_start = start
    while True:
        chunk_end = min(end, current_start + relativedelta(months=chunk_months))
        chunks.append((current_start, chunk_end))
        if chunk_end >= end:
            break
        current_start = current_start + relativedelta(months=step_months)
    return chunks
