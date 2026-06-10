"""CLI: pull Google Trends predictors at *daily* resolution into data/raw_daily/.

Same predictor universe as ``pull_trends.py`` but uses
``trends_client.pull_series(..., cadence="daily")`` so the API returns daily
bars. The 18-year window is split into ~96 overlapping 80-day chunks and
stitched. Budget: ~23,000 API calls × 5s pacing ≈ 32 hours of continuous
polling. Run as an overnight job with::

    caffeinate -di nohup python3 -u scripts/pull_trends_daily.py \\
        > pull_daily.log 2>&1 &

**Lid open, plugged in** — closing the laptop pauses everything.

The script is idempotent (already-cached chunks are read from disk on
restart) and resumable from any partial state.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from gtrends_bayes.config import PredictorsConfig
from gtrends_bayes.data.cache import cache_path, slugify
from gtrends_bayes.data.trends_client import (
    _build_pytrends_client,
    pull_series,
    validate_topic_mid,
)
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

DEFAULT_DAILY_RAW_ROOT = Path("data/raw_daily")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pull_trends_daily",
        description="Pull configured Trends predictors at daily resolution.",
    )
    p.add_argument("--config", default="config/predictors.yaml")
    p.add_argument("--geo", default=None)
    p.add_argument("--cache-root", default=str(DEFAULT_DAILY_RAW_ROOT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="restrict to the first N predictors (debug aid)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="override sampling.n_samples (default from config; "
                        "daily ingest may want to lower this to keep the "
                        "32-hour budget in line if rate-limits bite)")
    p.add_argument("--sleep-seconds", type=int, default=5,
                   help="pause between API calls (default 5s — daily ingest "
                        "uses tighter pacing than weekly's 60s because each "
                        "call returns less data per call)")
    p.add_argument("--chunk-days", type=int, default=80)
    p.add_argument("--overlap-days", type=int, default=10)
    p.add_argument("--skip-mid-validation", action="store_true")
    p.add_argument("--kinds", default="topic,category",
                   help="predictor kinds (default: 'topic,category')")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()

    cfg = PredictorsConfig.from_yaml(args.config)
    geo = args.geo or cfg.geo
    n_samples = args.n_samples if args.n_samples is not None else cfg.sampling.n_samples
    sleep_seconds = args.sleep_seconds
    cache_root = Path(args.cache_root)

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    unknown = kinds - {"topic", "category", "keyword"}
    if unknown:
        raise SystemExit(f"--kinds: unknown values {sorted(unknown)}")

    predictors = [p for p in cfg.predictors if p.kind in kinds]
    if args.limit:
        predictors = predictors[: args.limit]

    n_chunks_est = max(1, (cfg.window.end - cfg.window.start).days // (args.chunk_days - args.overlap_days))
    log.info(
        "DAILY pull: %d predictors × %d samples × ~%d chunks ≈ %d API calls "
        "(window=%s..%s, geo=%s, sleep=%ds, est=%.1fh)",
        len(predictors), n_samples, n_chunks_est,
        len(predictors) * n_samples * n_chunks_est, cfg.window.start,
        cfg.window.end, geo, sleep_seconds,
        len(predictors) * n_samples * n_chunks_est * sleep_seconds / 3600.0,
    )

    manifest_entries: list[dict] = []
    pytrends_client = None

    for pred in predictors:
        ident = pred.id if pred.kind == "category" else pred.mid
        slug = slugify(ident)
        stitched_existing = sum(
            cache_path(slug, geo, cfg.window.start, cfg.window.end, i, root=cache_root).exists()
            for i in range(n_samples)
        )
        entry = {
            "name": pred.name,
            "kind": pred.kind,
            "group": pred.group,
            "identifier": ident,
            "slug": slug,
            "samples_planned": n_samples,
            "samples_cached_before_run": stitched_existing,
        }

        if pred.kind == "topic" and not args.skip_mid_validation and not args.dry_run:
            ok = validate_topic_mid(pred.name, pred.mid)
            entry["mid_validated"] = ok
            if not ok:
                log.warning("skipping topic %r: mid %s no longer maps",
                            pred.name, pred.mid)
                entry["status"] = "skipped_mid_validation_failed"
                manifest_entries.append(entry)
                continue

        if args.dry_run:
            entry["status"] = ("dry_run_already_cached"
                               if stitched_existing == n_samples
                               else "dry_run_would_pull")
            manifest_entries.append(entry)
            continue

        if stitched_existing == n_samples:
            entry["status"] = "all_cached"
            manifest_entries.append(entry)
            log.info("[%s] all %d daily samples cached, skipping", pred.name, n_samples)
            continue

        if pytrends_client is None:
            pytrends_client = _build_pytrends_client()

        log.info("[%s] pulling daily (kind=%s, ident=%s)",
                 pred.name, pred.kind, ident)
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
            cadence="daily",
            chunk_days=args.chunk_days,
            overlap_days=args.overlap_days,
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
        "cadence": "daily",
        "window": {"start": cfg.window.start.isoformat(), "end": cfg.window.end.isoformat()},
        "n_samples": n_samples,
        "chunk_days": args.chunk_days,
        "overlap_days": args.overlap_days,
        "sleep_seconds": sleep_seconds,
        "entries": manifest_entries,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("wrote manifest to %s/_manifest.json", cache_root)

    n_skipped = sum(1 for e in manifest_entries if e["status"].startswith("skipped"))
    n_partial = sum(1 for e in manifest_entries if e["status"] == "partial")
    log.info(
        "done: %d entries (%d skipped, %d partial)",
        len(manifest_entries), n_skipped, n_partial,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
