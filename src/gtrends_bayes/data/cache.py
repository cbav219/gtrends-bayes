"""Disk cache for raw Trends pulls (Parquet under data/raw/).

Cache key: (query_slug, geo, start, end, sample_idx). Idempotent — if the
target Parquet exists, ``trends_client.pull_series`` skips the API call.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_RAW_ROOT = Path("data/raw")


def slugify(query: str | int) -> str:
    """Filesystem-safe slug for a query (str keyword/topic-mid or int category id)."""
    if isinstance(query, int):
        return f"cat_{query}"
    s = str(query).strip().lower()
    # Topic mids are like "/m/01jwbf"; flatten the slashes.
    s = s.replace("/", "_").lstrip("_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def cache_path(
    query_slug: str,
    geo: str,
    start: date,
    end: date,
    sample_idx: int,
    root: Path = DEFAULT_RAW_ROOT,
) -> Path:
    """Canonical Parquet path for a single (stitched) sample of a series."""
    return Path(root) / query_slug / geo / f"{start.isoformat()}_{end.isoformat()}" / f"sample_{sample_idx:02d}.parquet"


def chunk_cache_path(
    query_slug: str,
    geo: str,
    chunk_start: date,
    chunk_end: date,
    sample_idx: int,
    root: Path = DEFAULT_RAW_ROOT,
) -> Path:
    """Path for a single raw-chunk pull (one of K chunks comprising a stitched series)."""
    return (
        Path(root)
        / query_slug
        / geo
        / "chunks"
        / f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
        / f"sample_{sample_idx:02d}.parquet"
    )


def write_sample(df: pd.DataFrame, path: Path) -> None:
    """Write one sample DataFrame to Parquet (Snappy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def read_sample(path: Path) -> pd.DataFrame:
    """Read one cached sample DataFrame from Parquet."""
    return pd.read_parquet(path, engine="pyarrow")
