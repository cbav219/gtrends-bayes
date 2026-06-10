"""Stitch a sequence of weekly Trends chunks into one continuous series.

Each pytrends pull returns SVI normalized to 100 within that pull's date
window. To stitch chunks A and B into a single coherent series we rescale B
so its overlap mean matches A's overlap mean, then concatenate non-overlap
dates of B onto A. The procedure chains across an arbitrary number of chunks.

After all chunks are stitched, the result is rescaled so its overall max is
100 — restoring the standard SVI 0–100 convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

# If the overlap region averages below this, the rescale ratio is unstable;
# we fall back to using the chunk's raw values without rescaling and warn.
_MIN_OVERLAP_MEAN = 0.5
_MIN_OVERLAP_OBS = 4    # weeks (default — see `min_overlap_obs` parameter)


def stitch_chunks(
    chunks: list[pd.DataFrame],
    rescale_to_100: bool = True,
    min_overlap_obs: int = _MIN_OVERLAP_OBS,
) -> pd.DataFrame:
    """Stitch a list of overlapping chunks into one series.

    Parameters
    ----------
    chunks : list of pandas.DataFrame
        Each chunk is long-form with columns ``date | query | sample_idx | svi``,
        sorted by date. All chunks must share the same ``query`` and
        ``sample_idx`` value (the caller is responsible for passing per-sample
        chunk lists).
    rescale_to_100 : bool, default True
        If True, rescale the final stitched series so its max is 100.
    min_overlap_obs : int, default 4
        Minimum number of overlapping observations required to compute a
        rescale ratio. Defaults match weekly cadence (4 weeks). For daily
        ingest, callers should pass ``min_overlap_obs=5`` (≈1 business week)
        to require a stable overlap before rescaling.

    Returns
    -------
    pandas.DataFrame
        Single long-form dataframe with deduplicated dates spanning the union
        of all chunk windows.
    """
    if not chunks:
        return pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])

    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        return pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])

    if len(non_empty) == 1:
        out = non_empty[0].copy()
        if rescale_to_100 and out["svi"].max() > 0:
            out["svi"] = out["svi"] * (100.0 / out["svi"].max())
        return out.sort_values("date").reset_index(drop=True)

    # Index each chunk by date for overlap math, sorted ascending.
    sorted_chunks = sorted(non_empty, key=lambda c: c["date"].min())

    query = sorted_chunks[0]["query"].iloc[0]
    sample_idx = int(sorted_chunks[0]["sample_idx"].iloc[0])

    # Start from chunk 0; iteratively absorb each subsequent chunk.
    stitched = sorted_chunks[0].set_index("date")["svi"].astype(float).copy()

    for next_chunk in sorted_chunks[1:]:
        next_series = next_chunk.set_index("date")["svi"].astype(float)
        overlap_idx = stitched.index.intersection(next_series.index)

        if len(overlap_idx) < min_overlap_obs:
            log.warning(
                "stitch %s sample=%d: only %d overlapping observations between chunks "
                "(need >= %d) — appending without rescale",
                query, sample_idx, len(overlap_idx), min_overlap_obs,
            )
            scale = 1.0
        else:
            mean_left = stitched.loc[overlap_idx].mean()
            mean_right = next_series.loc[overlap_idx].mean()
            if mean_right < _MIN_OVERLAP_MEAN or not np.isfinite(mean_right):
                log.warning(
                    "stitch %s sample=%d: degenerate overlap mean (%.3f) — "
                    "appending without rescale",
                    query, sample_idx, mean_right,
                )
                scale = 1.0
            else:
                scale = float(mean_left / mean_right)

        scaled = next_series * scale
        # Keep stitched values for the overlap; append only new dates from next.
        new_dates = scaled.index.difference(stitched.index)
        stitched = pd.concat([stitched, scaled.loc[new_dates]]).sort_index()

    if rescale_to_100 and stitched.max() > 0:
        stitched = stitched * (100.0 / stitched.max())

    out = pd.DataFrame(
        {
            "date": stitched.index,
            "query": query,
            "sample_idx": sample_idx,
            "svi": stitched.values,
        }
    )
    return out.reset_index(drop=True)
