"""Thin wrapper around ``pytrends.request.TrendReq`` with multi-sampling and caching.

For windows longer than ~5 years pytrends returns *monthly* bars rather than
weekly, so ``pull_series`` automatically splits long windows into overlapping
chunks (see ``data/chunks.py``), pulls each chunk in weekly resolution, and
stitches them by overlap-mean rescaling (see ``data/stitch.py``). The stitched
output and per-chunk raw pulls are both cached so reruns avoid re-hitting the
API.

``pull_series`` is idempotent — already-cached samples (or chunks) are read
from disk and only the missing ones trigger API calls. Failed samples are
logged and skipped (returned DataFrame may have fewer than ``n_samples``
draws).
"""

from __future__ import annotations

import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

from gtrends_bayes.data.cache import (
    DEFAULT_RAW_ROOT,
    cache_path,
    chunk_cache_path,
    read_sample,
    slugify,
    write_sample,
)
from gtrends_bayes.data.chunks import (
    DEFAULT_CHUNK_MONTHS,
    DEFAULT_OVERLAP_MONTHS,
    make_chunks,
)
from gtrends_bayes.data.chunks_daily import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_OVERLAP_DAYS,
    make_chunks_daily,
)
from gtrends_bayes.data.stitch import stitch_chunks
from gtrends_bayes.logging import get_logger

QueryKind = Literal["keyword", "category", "topic"]
Cadence = Literal["weekly", "daily"]

log = get_logger(__name__)


def _build_pytrends_client(timeout: tuple[int, int] = (10, 30)):
    """Construct a TrendReq client; isolated for monkeypatching in tests."""
    from pytrends.request import TrendReq

    return TrendReq(hl="en-US", tz=0, timeout=timeout)


def _build_payload(
    pytrends_client,  # noqa: ANN001
    query: str | int,
    kind: QueryKind,
    geo: str,
    start: date,
    end: date,
) -> None:
    """Configure the pytrends client for a single query."""
    timeframe = f"{start.isoformat()} {end.isoformat()}"
    if kind == "category":
        cat = int(query)
        pytrends_client.build_payload(kw_list=[""], cat=cat, timeframe=timeframe, geo=geo)
    elif kind == "topic":
        # Topic mids are passed in kw_list; pytrends treats them as encoded keys.
        pytrends_client.build_payload(kw_list=[str(query)], cat=0, timeframe=timeframe, geo=geo)
    elif kind == "keyword":
        pytrends_client.build_payload(kw_list=[str(query)], cat=0, timeframe=timeframe, geo=geo)
    else:
        raise ValueError(f"unknown kind: {kind!r}")


def _fetch_one_sample(
    pytrends_client,  # noqa: ANN001
    query: str | int,
    kind: QueryKind,
    geo: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Pull one draw of the SVI series for the (start, end) window.

    Returns long-form ``date | query | svi`` (sample_idx filled by caller).
    """
    _build_payload(pytrends_client, query, kind, geo, start, end)
    raw = pytrends_client.interest_over_time()
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "query", "svi"])
    if "isPartial" in raw.columns:
        raw = raw.drop(columns=["isPartial"])
    value_col = raw.columns[0]
    out = (
        raw[[value_col]]
        .rename(columns={value_col: "svi"})
        .reset_index()
    )
    out["query"] = str(query)
    return out[["date", "query", "svi"]]


# Default backoff schedule for 429 / rate-limit errors.
# Extended 2026-05-12 for the v3 daily pull: Google's quota-reset window can
# be 20+ minutes during sustained throttling. The extra retries let us wait
# it out instead of giving up after 7 minutes and dropping the chunk.
# Total worst-case wait per chunk: 60+120+240+600+1200 = ~37 minutes.
_DEFAULT_BACKOFF_SECONDS: tuple[int, ...] = (60, 120, 240, 600, 1200)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic: pytrends wraps Google's 429 in plain Python exceptions whose
    message includes the status code. Match that without depending on a
    particular exception class."""
    msg = str(exc)
    return "429" in msg or "rate" in msg.lower() and "limit" in msg.lower()


def _fetch_with_retry(
    pytrends_client,  # noqa: ANN001
    query: str | int,
    kind: QueryKind,
    geo: str,
    start: date,
    end: date,
    backoff_seconds: tuple[int, ...] = _DEFAULT_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Wrap ``_fetch_one_sample`` with retries on rate-limit (429) errors only.

    Other exceptions propagate immediately to the caller, which logs them and
    skips the chunk. Returns ``_fetch_one_sample``'s output on success.
    """
    attempts = len(backoff_seconds) + 1
    for attempt in range(attempts):
        try:
            return _fetch_one_sample(pytrends_client, query, kind, geo, start, end)
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit_error(exc) or attempt >= attempts - 1:
                raise
            wait = backoff_seconds[attempt]
            log.warning(
                "rate-limited fetching %s (%s..%s); backing off %ds (attempt %d/%d)",
                query, start, end, wait, attempt + 1, attempts - 1,
            )
            time.sleep(wait)
    # Unreachable, but pacifies type checkers.
    raise RuntimeError("retry loop exited without returning or raising")


def _pull_chunked_sample(
    query: str | int,
    kind: QueryKind,
    geo: str,
    chunks: list[tuple[date, date]],
    sample_idx: int,
    sleep_seconds: int,
    cache_root: Path,
    pytrends_client,  # noqa: ANN001
    inter_call_state: dict,
    min_overlap_obs: int = 4,
) -> tuple[pd.DataFrame, object]:
    """Fetch (or read from cache) every chunk for one sample, then stitch.

    ``inter_call_state`` is a single-item dict carrying ``{"client": ..., "calls_made": int}``
    so consecutive calls across samples share rate-limit accounting.
    """
    slug = slugify(query)
    chunk_frames: list[pd.DataFrame] = []
    for cs, ce in chunks:
        cpath = chunk_cache_path(slug, geo, cs, ce, sample_idx, root=cache_root)
        if cpath.exists():
            log.debug("chunk cache hit %s", cpath)
            chunk_frames.append(read_sample(cpath))
            continue

        if pytrends_client is None:
            pytrends_client = _build_pytrends_client()
            inter_call_state["client"] = pytrends_client
        if inter_call_state.get("calls_made", 0) > 0 and sleep_seconds:
            log.debug("sleeping %ds before next API call", sleep_seconds)
            time.sleep(sleep_seconds)
        try:
            df = _fetch_with_retry(pytrends_client, query, kind, geo, cs, ce)
        except Exception as exc:  # noqa: BLE001 — pytrends throws ad-hoc exceptions
            log.warning(
                "pytrends fetch failed for %s sample %d chunk (%s..%s): %s",
                query, sample_idx, cs, ce, exc,
            )
            inter_call_state["calls_made"] = inter_call_state.get("calls_made", 0) + 1
            continue

        inter_call_state["calls_made"] = inter_call_state.get("calls_made", 0) + 1
        if df.empty:
            log.warning(
                "empty SVI for %s sample %d chunk (%s..%s, geo=%s)",
                query, sample_idx, cs, ce, geo,
            )
            continue
        df = df.copy()
        df["sample_idx"] = sample_idx
        df = df[["date", "query", "sample_idx", "svi"]]
        write_sample(df, cpath)
        chunk_frames.append(df)

    return stitch_chunks(chunk_frames, min_overlap_obs=min_overlap_obs), pytrends_client


def pull_series(
    query: str | int,
    kind: QueryKind,
    geo: str,
    start: date,
    end: date,
    n_samples: int = 6,
    sleep_seconds: int = 60,
    cache_root: Path = DEFAULT_RAW_ROOT,
    pytrends_client=None,  # noqa: ANN001
    chunk_months: int = DEFAULT_CHUNK_MONTHS,
    overlap_months: int = DEFAULT_OVERLAP_MONTHS,
    cadence: Cadence = "weekly",
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> pd.DataFrame:
    """Pull a Google Trends series ``n_samples`` times, returning a long-form frame.

    For windows longer than ``chunk_months`` (default 4.5 years), the pull is
    transparently chunked and stitched so the output is at weekly resolution.
    Per-chunk and stitched-output Parquet caches are populated under
    ``cache_root``; reruns read from cache instead of hitting the API.

    Parameters
    ----------
    query : str or int
        Keyword string, category id (int), or topic mid (e.g. ``"/m/01jwbf"``).
    kind : {"keyword", "category", "topic"}
    geo : str
        ISO geo code, e.g. ``"US"``.
    start, end : date
    n_samples : int, default 6
    sleep_seconds : int, default 60
        Pause inserted between consecutive *new* API calls (cache hits skip
        the sleep). Counts across chunks and samples — the entire pull
        respects one global cadence.
    cache_root : Path, default ``data/raw``
    pytrends_client : optional
        A pre-built TrendReq instance. If None, one is constructed lazily on
        the first non-cached call.
    chunk_months : int, default 54 (4.5y)
    overlap_months : int, default 12 (1y)
    cadence : {"weekly", "daily"}, default "weekly"
        ``"weekly"`` uses the month-scale chunker (default 4.5y windows).
        ``"daily"`` switches to the day-scale chunker (80-day windows, 10-day
        overlap) — needed for the v3 daily-resolution ingest pipeline. Caller
        is responsible for using a separate ``cache_root`` for daily pulls
        (e.g. ``data/raw_daily``) so weekly and daily caches don't collide.
    chunk_days : int, default 80
        Daily-chunker window size. Only used when ``cadence == "daily"``.
    overlap_days : int, default 10
        Daily-chunker overlap. Only used when ``cadence == "daily"``.

    Returns
    -------
    pandas.DataFrame
        Long format ``date | query | sample_idx | svi``. May contain fewer
        than ``n_samples`` distinct ``sample_idx`` values if some samples
        failed entirely.
    """
    slug = slugify(query)
    if cadence == "weekly":
        chunks = make_chunks(
            start, end, chunk_months=chunk_months, overlap_months=overlap_months,
        )
    elif cadence == "daily":
        chunks = make_chunks_daily(
            start, end, chunk_days=chunk_days, overlap_days=overlap_days,
        )
    else:
        raise ValueError(f"cadence={cadence!r} not in ('weekly', 'daily')")
    # Stable rescale needs ~1 business week of overlap at daily cadence;
    # 4 weeks at weekly cadence is the historical default.
    min_overlap_obs = 5 if cadence == "daily" else 4
    log.debug(
        "pull_series %s (kind=%s, cadence=%s): %d chunk(s) over %s..%s",
        query, kind, cadence, len(chunks), start, end,
    )

    frames: list[pd.DataFrame] = []
    inter_call_state = {"client": pytrends_client, "calls_made": 0}

    for sample_idx in range(n_samples):
        stitched_path = cache_path(slug, geo, start, end, sample_idx, root=cache_root)
        if stitched_path.exists():
            log.debug("stitched cache hit %s", stitched_path)
            frames.append(read_sample(stitched_path))
            continue

        if len(chunks) == 1:
            # Single-window path — keep prior behavior, no stitching needed.
            cs, ce = chunks[0]
            if pytrends_client is None:
                pytrends_client = _build_pytrends_client()
                inter_call_state["client"] = pytrends_client
            if inter_call_state.get("calls_made", 0) > 0 and sleep_seconds:
                log.debug("sleeping %ds before next API call", sleep_seconds)
                time.sleep(sleep_seconds)
            try:
                df = _fetch_with_retry(pytrends_client, query, kind, geo, cs, ce)
            except Exception as exc:  # noqa: BLE001
                log.warning("pytrends fetch failed for %s sample %d: %s",
                            query, sample_idx, exc)
                inter_call_state["calls_made"] = inter_call_state.get("calls_made", 0) + 1
                continue
            inter_call_state["calls_made"] = inter_call_state.get("calls_made", 0) + 1
            if df.empty:
                log.warning("empty SVI for %s sample %d (geo=%s, %s..%s)",
                            query, sample_idx, geo, cs, ce)
                continue
            df = df.copy()
            df["sample_idx"] = sample_idx
            df = df[["date", "query", "sample_idx", "svi"]]
            write_sample(df, stitched_path)
            frames.append(df)
        else:
            # Chunked + stitched path.
            stitched, pytrends_client = _pull_chunked_sample(
                query=query,
                kind=kind,
                geo=geo,
                chunks=chunks,
                sample_idx=sample_idx,
                sleep_seconds=sleep_seconds,
                cache_root=cache_root,
                pytrends_client=pytrends_client,
                inter_call_state=inter_call_state,
                min_overlap_obs=min_overlap_obs,
            )
            if stitched.empty:
                continue
            write_sample(stitched, stitched_path)
            frames.append(stitched)

    if not frames:
        return pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=512)
def _suggestions_for(name: str) -> tuple[tuple[str, str, str], ...]:
    """Memoized wrapper around pytrends.suggestions; returns tuple of (mid, title, type)."""
    client = _build_pytrends_client()
    suggestions = client.suggestions(keyword=name)
    return tuple((s.get("mid", ""), s.get("title", ""), s.get("type", "")) for s in suggestions)


def validate_topic_mid(name: str, mid: str, sleep_seconds: int = 5) -> bool:
    """Confirm a topic ``mid`` still maps to ``name`` via ``pytrends.suggestions``.

    Returns True iff any suggestion for ``name`` has matching mid AND a
    non-empty type. A short sleep is inserted to avoid hammering the
    suggestions endpoint.
    """
    try:
        suggestions = _suggestions_for(name)
    except Exception as exc:  # noqa: BLE001
        log.warning("suggestions lookup failed for %s: %s", name, exc)
        return False
    finally:
        if sleep_seconds:
            time.sleep(sleep_seconds)
    for mid_, _title, type_ in suggestions:
        if mid_ == mid and type_:
            return True
    log.warning("topic mid %s for %r not found in suggestions: %s",
                mid, name, [(m, t) for m, _, t in suggestions])
    return False
