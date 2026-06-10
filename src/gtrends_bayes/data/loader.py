"""Bridge cached raw Trends pulls to the preprocessing Pipeline.

``load_predictor_samples`` walks the predictors config, reads every stitched
sample from ``data/raw/{slug}/{geo}/{start}_{end}/sample_*.parquet`` it can
find, optionally renames the ``query`` column to the human-readable predictor
name, and returns a single long-form DataFrame that ``Pipeline.fit_transform``
can consume directly.

Predictors that have no cached samples yet (the pull is still running for
them) are silently skipped — pair the loader with ``predictor_classes`` so the
Pipeline's ``classes`` map only references columns that actually appear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from gtrends_bayes.config import PredictorEntry, PredictorsConfig
from gtrends_bayes.data.cache import DEFAULT_RAW_ROOT, slugify
from gtrends_bayes.logging import get_logger

QueryClass = Literal["category", "topic"]

log = get_logger(__name__)


def _identifier(pred: PredictorEntry) -> str | int:
    return pred.id if pred.kind == "category" else pred.mid


def _stitched_dir(pred: PredictorEntry, geo: str, cfg: PredictorsConfig,
                  cache_root: Path) -> Path:
    slug = slugify(_identifier(pred))
    return (
        cache_root / slug / geo
        / f"{cfg.window.start.isoformat()}_{cfg.window.end.isoformat()}"
    )


def load_predictor_samples(
    cfg: PredictorsConfig,
    cache_root: Path = DEFAULT_RAW_ROOT,
    geo: str | None = None,
    rename_to_human: bool = True,
) -> pd.DataFrame:
    """Concatenate every cached stitched sample into one long-form DataFrame.

    Parameters
    ----------
    cfg : PredictorsConfig
        Loaded ``config/predictors.yaml``.
    cache_root : Path, default ``data/raw``
    geo : str, optional
        Override ``cfg.geo`` (e.g. for multi-geo experiments).
    rename_to_human : bool, default True
        If True, the ``query`` column holds the predictor's friendly name
        (e.g. "Jobs", "Economic crisis"). If False, the raw cache identifier
        ("60", "/m/01jwbf") is preserved.

    Returns
    -------
    pandas.DataFrame
        Long format ``date | query | sample_idx | svi``, ready to feed
        ``Pipeline.fit_transform``. Empty if no predictors are cached yet.
    """
    geo = geo or cfg.geo
    frames: list[pd.DataFrame] = []
    for pred in cfg.predictors:
        sdir = _stitched_dir(pred, geo, cfg, cache_root)
        if not sdir.exists():
            log.debug("no cached samples for %s (%s) at %s", pred.name, _identifier(pred), sdir)
            continue
        sample_paths = sorted(sdir.glob("sample_*.parquet"))
        if not sample_paths:
            log.debug("no sample_*.parquet under %s", sdir)
            continue
        for p in sample_paths:
            df = pd.read_parquet(p)
            if rename_to_human:
                df = df.copy()
                df["query"] = pred.name
            frames.append(df)
    if not frames:
        log.warning("no cached predictor samples found under %s", cache_root)
        return pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def predictor_classes(
    cfg: PredictorsConfig,
    rename_to_human: bool = True,
) -> dict[str, QueryClass]:
    """Map each predictor's column name to ``"category"`` or ``"topic"``.

    Use the same ``rename_to_human`` setting as the matching
    ``load_predictor_samples`` call so the keys here line up with the columns
    in the wide DataFrame the Pipeline produces.
    """
    out: dict[str, QueryClass] = {}
    for pred in cfg.predictors:
        key = pred.name if rename_to_human else str(_identifier(pred))
        out[key] = pred.kind  # type: ignore[assignment]
    return out
