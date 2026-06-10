"""CLI: read config/predictors.yaml and pull every entry into data/raw/.

Usage examples
--------------
    # Plan only — no API calls, no writes
    python scripts/pull_trends.py --dry-run

    # Try a small subset (first 3 predictors, 2 samples each)
    python scripts/pull_trends.py --limit 3 --n-samples 2

    # Full run (long — multi_sampling × ~25 predictors × 60-second sleeps)
    python scripts/pull_trends.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from gtrends_bayes.config import PredictorEntry, PredictorsConfig
from gtrends_bayes.data.cache import DEFAULT_RAW_ROOT, cache_path, slugify
from gtrends_bayes.data.trends_client import (
    _build_pytrends_client,
    pull_series,
    validate_topic_mid,
)
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pull_trends",
        description="Pull configured Google Trends predictors into data/raw/.",
    )
    p.add_argument("--config", default="config/predictors.yaml", help="path to predictors YAML")
    p.add_argument("--geo", default=None, help="override geo (default from config)")
    p.add_argument("--cache-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--dry-run", action="store_true",
                   help="plan only; no API calls or writes")
    p.add_argument("--limit", type=int, default=None,
                   help="restrict to the first N predictors (debug aid)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="override sampling.n_samples (debug aid)")
    p.add_argument("--sleep-seconds", type=int, default=None,
                   help="override sampling.sleep_seconds")
    p.add_argument("--skip-mid-validation", action="store_true",
                   help="don't call pytrends.suggestions for topic-mid sanity check")
    p.add_argument("--kinds", default="topic,category",
                   help="comma-separated predictor kinds to pull "
                        "(default: 'topic,category'; pass 'topic' for topics-only)")
    return p


def _planned_samples(
    pred: PredictorEntry,
    geo: str,
    start,
    end,
    n_samples: int,
    cache_root: Path,
) -> tuple[int, int]:
    """Return (cached, missing) count for a predictor."""
    slug = slugify(pred.id if pred.kind == "category" else pred.mid)
    cached = sum(
        cache_path(slug, geo, start, end, i, root=cache_root).exists()
        for i in range(n_samples)
    )
    return cached, n_samples - cached


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()

    cfg = PredictorsConfig.from_yaml(args.config)
    geo = args.geo or cfg.geo
    n_samples = args.n_samples if args.n_samples is not None else cfg.sampling.n_samples
    sleep_seconds = (
        args.sleep_seconds if args.sleep_seconds is not None else cfg.sampling.sleep_seconds
    )
    cache_root = Path(args.cache_root)

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    unknown = kinds - {"topic", "category", "keyword"}
    if unknown:
        raise SystemExit(f"--kinds: unknown values {sorted(unknown)}")

    predictors = [p for p in cfg.predictors if p.kind in kinds]
    if args.limit:
        predictors = predictors[: args.limit]
    log.info(
        "loaded %d predictors from %s (kinds=%s, geo=%s, window=%s..%s, "
        "n_samples=%d, dry_run=%s)",
        len(predictors), args.config, sorted(kinds), geo,
        cfg.window.start, cfg.window.end, n_samples, args.dry_run,
    )

    manifest_entries: list[dict] = []
    pytrends_client = None  # constructed lazily on first non-cached call

    for pred in predictors:
        ident = pred.id if pred.kind == "category" else pred.mid
        slug = slugify(ident)
        cached, missing = _planned_samples(
            pred, geo, cfg.window.start, cfg.window.end, n_samples, cache_root
        )
        entry = {
            "name": pred.name,
            "kind": pred.kind,
            "group": pred.group,
            "identifier": ident,
            "slug": slug,
            "samples_planned": n_samples,
            "samples_cached_before_run": cached,
        }

        if pred.kind == "topic" and not args.skip_mid_validation and not args.dry_run:
            ok = validate_topic_mid(pred.name, pred.mid)
            entry["mid_validated"] = ok
            if not ok:
                log.warning("skipping topic %r: mid %s no longer maps to that name",
                            pred.name, pred.mid)
                entry["status"] = "skipped_mid_validation_failed"
                manifest_entries.append(entry)
                continue

        if args.dry_run:
            entry["status"] = "dry_run_would_pull" if missing else "dry_run_already_cached"
            manifest_entries.append(entry)
            continue

        if missing == 0:
            entry["status"] = "all_cached"
            manifest_entries.append(entry)
            log.info("[%s] all %d samples cached, skipping", pred.name, n_samples)
            continue

        if pytrends_client is None:
            pytrends_client = _build_pytrends_client()

        log.info("[%s] pulling %d new samples (kind=%s, ident=%s)",
                 pred.name, missing, pred.kind, ident)
        df = pull_series(
            query=ident,
            kind=pred.kind,
            geo=geo,
            start=cfg.window.start,
            end=cfg.window.end,
            n_samples=n_samples,
            sleep_seconds=sleep_seconds,
            cache_root=cache_root,
            pytrends_client=pytrends_client,
        )
        cached_after = sum(
            cache_path(slug, geo, cfg.window.start, cfg.window.end, i, root=cache_root).exists()
            for i in range(n_samples)
        )
        entry["samples_cached_after_run"] = cached_after
        entry["rows_returned"] = len(df)
        entry["status"] = "ok" if cached_after == n_samples else "partial"
        manifest_entries.append(entry)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "geo": geo,
        "window": {"start": cfg.window.start.isoformat(), "end": cfg.window.end.isoformat()},
        "n_samples": n_samples,
        "dry_run": args.dry_run,
        "entries": manifest_entries,
    }

    manifest_path = cache_root / "_manifest.json"
    if args.dry_run:
        log.info("dry-run manifest (not written):\n%s", json.dumps(manifest, indent=2, default=str))
    else:
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        log.info("wrote manifest to %s", manifest_path)

    n_skipped = sum(1 for e in manifest_entries if e["status"].startswith("skipped"))
    n_partial = sum(1 for e in manifest_entries if e["status"] == "partial")
    log.info(
        "done: %d entries (%d skipped, %d partial)",
        len(manifest_entries), n_skipped, n_partial,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
