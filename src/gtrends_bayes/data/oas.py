"""ICE BofA HY / IG OAS daily history via WRDS, with FRED stitch fallback.

.. warning::

   **DEFERRED — not active in v4/v5.** v3 Phase A (OAS swap) was deferred
   2026-05-12 after the UChicago WRDS investigation confirmed the
   subscription does NOT carry the aggregate ICE BofA index series (only
   individual-bond data in ``wrdsapps_bondret.bondret``). See
   ``IMPLEMENTATION_PLAN_v3.md`` §A and
   ``project_wrds_oas_not_available.md`` in the Claude memory.

   This module ships as a working stub in case (a) a Bloomberg seat
   materializes or (b) a paid Nasdaq Data Link subscription is added.
   Until then, the project uses HYG / LQD ETF proxies (see
   ``targets.yaml``), and the v4/v5 bundle deliberately **excludes this
   file** (see ``scripts/build_v4_bundle.py`` — only ``inference/`` and
   the modules in ``_BUNDLED_SUPPORT_MODULES`` get shipped).

   The 2023+ FRED OAS CSVs the user dropped at ``data/csv/`` are
   processed by ``scripts/oas_overlay_v3.py`` for the PM reference
   overlay — that path does NOT use this module.

FRED's ``BAMLH0A0HYM2`` (HY) and ``BAMLC0A0CM`` (IG) only go back to
2023-05-02, too short for the locked 2008+ training window. WRDS carries
the full ICE BofA index history (back to 1996-12-31 daily) **in
principle** — but the UChicago subscription does not. v3 prefers WRDS as
the primary source and uses FRED as a post-2023 cross-check via
``stitch`` — both behaviors below are correct, just gated on having WRDS
access to the right library.

The WRDS index library / table name is discovered at runtime — the ICE
BofA indices live under one of several candidate libraries
(``ice_baml``, ``bofaml``, ``wrdsapps_windices``, …) depending on the
WRDS subscription. ``fetch_oas`` tries the candidates in order, picking
the first that has a table containing both index codes.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from gtrends_bayes.logging import get_logger

Target = Literal["HY", "IG"]
Source = Literal["wrds", "fred", "stitch"]

log = get_logger(__name__)

# ICE BofA index codes — the spec is BAMLH0A0HYM2 (HY) and BAMLC0A0CM (IG)
# on FRED; the corresponding ICE codes used inside WRDS BAML tables are
# H0A0 (HY US High Yield Master II) and C0A0 (IG US Corporate Master).
ICE_INDEX_CODE: dict[Target, str] = {"HY": "H0A0", "IG": "C0A0"}
FRED_TICKER: dict[Target, str] = {"HY": "BAMLH0A0HYM2", "IG": "BAMLC0A0CM"}

# WRDS schemas that have historically carried ICE BofA OAS series. The first
# entry that ``conn.list_tables`` reports as having a recognizable OAS table
# wins. Callers can override via ``wrds_library``.
_WRDS_LIBRARY_CANDIDATES: tuple[str, ...] = (
    "ice_baml",
    "bofaml",
    "wrdsapps_windices",
    "fixedincome",
)

# Date overlap window we use to verify WRDS / FRED agreement when stitching.
_STITCH_OVERLAP_START = pd.Timestamp("2023-05-02")
_STITCH_OVERLAP_END = pd.Timestamp("2023-12-31")
_STITCH_MAX_DIVERGENCE_BPS = 5.0  # absolute mean-diff over overlap

DEFAULT_RAW_OAS_ROOT = Path("data/raw_oas")


def _wrds_connect():
    """Connect to WRDS using credentials from the .env file.

    Reads ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` from the environment (the
    project's ``.env`` is loaded by the caller / pytest fixture). The wrds
    package picks up the password from ``PGPASSWORD``.
    """
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise RuntimeError(
            "WRDS_USERNAME not set. Add WRDS_USERNAME + WRDS_PASSWORD to .env "
            "(see .env.example)."
        )
    password = os.environ.get("WRDS_PASSWORD")
    if password and not os.environ.get("PGPASSWORD"):
        os.environ["PGPASSWORD"] = password

    import wrds  # imported lazily so tests can monkeypatch

    return wrds.Connection(wrds_username=user)


def _resolve_wrds_table(conn, library: str | None = None) -> tuple[str, str]:
    """Return ``(library, table)`` for ICE BofA OAS series.

    If ``library`` is given, only that library is probed; otherwise the
    module-level candidate list is walked in order. Raises if no candidate
    library has a table whose name contains ``oas`` (case-insensitive).
    """
    candidates = (library,) if library else _WRDS_LIBRARY_CANDIDATES
    for lib in candidates:
        try:
            tables = conn.list_tables(library=lib)
        except Exception as exc:  # noqa: BLE001 — wrds raises ad-hoc errors
            log.debug("WRDS library %s not accessible: %s", lib, exc)
            continue
        oas_tables = [t for t in tables if "oas" in t.lower() or "spread" in t.lower()]
        if oas_tables:
            return lib, oas_tables[0]
    raise RuntimeError(
        f"No WRDS library among {candidates} contains an OAS / spread table. "
        "Probe `conn.list_libraries()` manually and pass `wrds_library=...` "
        "to fetch_oas()."
    )


def _fetch_oas_wrds(
    target: Target,
    start: date,
    end: date,
    wrds_library: str | None = None,
    wrds_table: str | None = None,
) -> pd.Series:
    """Pull daily OAS in bps from WRDS for the requested ICE index code."""
    code = ICE_INDEX_CODE[target]
    conn = _wrds_connect()
    try:
        if wrds_table is None:
            wrds_library, wrds_table = _resolve_wrds_table(conn, wrds_library)
        log.info(
            "fetching OAS via WRDS %s.%s (index=%s, %s..%s)",
            wrds_library, wrds_table, code, start, end,
        )
        # We don't pre-know the table's column names; query the first row to
        # discover them, then build the actual pull. The expected schema is
        # something like (date, index_code, oas, …) — we accept any column
        # named like 'date'/'caldt' for the date and 'oas'/'spread' for value.
        probe = conn.raw_sql(
            f"SELECT * FROM {wrds_library}.{wrds_table} LIMIT 1"
        )
        cols = {c.lower(): c for c in probe.columns}
        date_col = next((cols[k] for k in ("date", "caldt", "datadate") if k in cols), None)
        val_col = next(
            (cols[k] for k in ("oas", "spread", "oas_bps", "value") if k in cols),
            None,
        )
        idx_col = next(
            (cols[k] for k in ("index_code", "index", "ticker", "code") if k in cols),
            None,
        )
        if not (date_col and val_col and idx_col):
            raise RuntimeError(
                f"Could not identify date / value / index columns in "
                f"{wrds_library}.{wrds_table}; got {list(probe.columns)}"
            )

        sql = (
            f"SELECT {date_col} AS date, {val_col} AS oas_bps "
            f"FROM {wrds_library}.{wrds_table} "
            f"WHERE {idx_col} = '{code}' "
            f"AND {date_col} >= '{start.isoformat()}' "
            f"AND {date_col} <= '{end.isoformat()}' "
            f"ORDER BY {date_col}"
        )
        df = conn.raw_sql(sql)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            f"WRDS returned no rows for index={code} in {start}..{end}; "
            "verify the index code and library/table names."
        )
    s = pd.Series(
        df["oas_bps"].astype(float).values,
        index=pd.to_datetime(df["date"]),
        name=f"{target}_OAS",
    )
    return s.dropna()


def _fetch_oas_fred(target: Target, start: date, end: date) -> pd.Series:
    """Pull daily OAS in bps from FRED (only ~2023+ history available)."""
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")
    ticker = FRED_TICKER[target]
    log.info("fetching OAS via FRED %s (%s..%s)", ticker, start, end)
    s = Fred(api_key=api_key).get_series(
        ticker, observation_start=start, observation_end=end,
    )
    s = s.rename(f"{target}_OAS")
    s.index = pd.to_datetime(s.index)
    # FRED reports OAS in *percentage points* (e.g., 3.87), not bps.
    # Multiply by 100 to align with WRDS units.
    return (s * 100.0).dropna()


def _stitch_oas(wrds_series: pd.Series, fred_series: pd.Series) -> pd.Series:
    """Splice WRDS history with FRED tail, verifying agreement at the overlap.

    Returns a single OAS series in bps. Raises if mean abs divergence over
    the overlap exceeds ``_STITCH_MAX_DIVERGENCE_BPS``.
    """
    overlap = wrds_series.index.intersection(fred_series.index)
    overlap = overlap[
        (overlap >= _STITCH_OVERLAP_START) & (overlap <= _STITCH_OVERLAP_END)
    ]
    if len(overlap) > 5:
        diff = (wrds_series.loc[overlap] - fred_series.loc[overlap]).abs().mean()
        log.info("WRDS vs FRED overlap: n=%d, mean |Δ|=%.2f bps", len(overlap), diff)
        if diff > _STITCH_MAX_DIVERGENCE_BPS:
            raise RuntimeError(
                f"WRDS vs FRED OAS diverged by {diff:.2f} bps over the "
                f"{_STITCH_OVERLAP_START.date()}..{_STITCH_OVERLAP_END.date()} "
                f"overlap (threshold {_STITCH_MAX_DIVERGENCE_BPS} bps). "
                "Investigate before trusting either source."
            )
    else:
        log.warning(
            "WRDS / FRED overlap too small (n=%d) to verify; using WRDS verbatim",
            len(overlap),
        )

    # Prefer WRDS where both have data; use FRED only for the post-WRDS tail.
    cutoff = wrds_series.index.max()
    fred_tail = fred_series[fred_series.index > cutoff]
    out = pd.concat([wrds_series, fred_tail]).sort_index()
    out.name = wrds_series.name
    return out


def _cache_path(target: Target, source: Source, start: date, end: date,
                root: Path = DEFAULT_RAW_OAS_ROOT) -> Path:
    return Path(root) / f"{target}_OAS_{source}_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_oas(
    target: Target,
    start: date,
    end: date,
    source: Source = "wrds",
    cache_root: Path = DEFAULT_RAW_OAS_ROOT,
    use_cache: bool = True,
    wrds_library: str | None = None,
    wrds_table: str | None = None,
) -> pd.Series:
    """Fetch HY or IG OAS daily series in basis points.

    Parameters
    ----------
    target : {"HY", "IG"}
    start, end : date
    source : {"wrds", "fred", "stitch"}, default "wrds"
        ``"wrds"`` uses ICE BofA index codes H0A0 / C0A0 (full history).
        ``"fred"`` uses BAMLH0A0HYM2 / BAMLC0A0CM (only ~2023-05 onwards).
        ``"stitch"`` uses WRDS history with FRED tail, verifying agreement
        over the overlap window (2023-05 → 2023-12).
    cache_root : Path
        Where to cache the resulting parquet.
    use_cache : bool, default True
        If True and a cached parquet exists, read from disk instead of hitting
        the upstream source.
    wrds_library, wrds_table : str, optional
        Override WRDS schema discovery (only used when ``source`` involves
        WRDS).

    Returns
    -------
    pandas.Series
        Daily OAS in bps, indexed by trading-day ``DatetimeIndex``, named
        ``"{target}_OAS"``.
    """
    cache = _cache_path(target, source, start, end, root=cache_root)
    if use_cache and cache.exists():
        log.info("OAS cache hit %s", cache)
        s = pd.read_parquet(cache).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s

    if source == "wrds":
        series = _fetch_oas_wrds(target, start, end, wrds_library, wrds_table)
    elif source == "fred":
        series = _fetch_oas_fred(target, start, end)
    elif source == "stitch":
        wrds_s = _fetch_oas_wrds(target, start, end, wrds_library, wrds_table)
        fred_s = _fetch_oas_fred(target, max(start, _STITCH_OVERLAP_START.date()), end)
        series = _stitch_oas(wrds_s, fred_s)
    else:
        raise ValueError(f"source={source!r} not in {('wrds','fred','stitch')}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(cache, engine="pyarrow", compression="snappy")
    log.info("OAS cached to %s (n=%d obs)", cache, len(series))
    return series
