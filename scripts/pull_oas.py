"""CLI: fetch HY/IG OAS daily history via WRDS, cache to data/raw_oas/.

Reads target rows with ``source=wrds`` from ``config/targets.yaml`` and pulls
each via ``data.oas.fetch_oas``. Idempotent — re-running with cached parquets
in place is a no-op. Sanity-checks: the returned series should span the
configured window and contain visible peaks at the GFC, Eurozone, COVID,
gilt, and SVB crises (printed if ``--sanity-check`` is set).

Usage::

    cd gtrends-bayes
    PYTHONPATH=src python scripts/pull_oas.py            # WRDS only
    PYTHONPATH=src python scripts/pull_oas.py --source stitch
    PYTHONPATH=src python scripts/pull_oas.py --target HY_OAS --sanity-check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from gtrends_bayes.config import TargetsConfig
from gtrends_bayes.data.oas import DEFAULT_RAW_OAS_ROOT, fetch_oas
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

# Crisis windows we expect to see as visible OAS spikes — used by --sanity-check
# to print the running max within each window for quick eyeballing.
_CRISIS_WINDOWS = [
    ("GFC", "2008-09-01", "2009-03-31"),
    ("Eurozone", "2011-08-01", "2012-06-30"),
    ("Oil-2016", "2015-12-01", "2016-02-29"),
    ("COVID", "2020-02-15", "2020-04-30"),
    ("Gilt", "2022-09-01", "2022-10-31"),
    ("SVB", "2023-03-01", "2023-04-30"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pull_oas")
    p.add_argument("--config", default="config/targets.yaml")
    p.add_argument("--source", choices=["wrds", "fred", "stitch"], default="wrds")
    p.add_argument(
        "--target", default=None,
        help="Pull only this target (HY_OAS / IG_OAS); default = all WRDS targets.",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_RAW_OAS_ROOT))
    p.add_argument("--no-cache", action="store_true",
                   help="Force a re-pull even if cache exists.")
    p.add_argument("--sanity-check", action="store_true",
                   help="Print max OAS within known crisis windows for visual verification.")
    p.add_argument("--wrds-library", default=None,
                   help="Override WRDS schema discovery (e.g. 'bofaml').")
    p.add_argument("--wrds-table", default=None,
                   help="Override WRDS table name within the library.")
    return p


def _sanity_check(series: pd.Series) -> dict:
    """Return per-window max OAS for visual sanity-checking."""
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    out = {}
    for label, lo, hi in _CRISIS_WINDOWS:
        sl = series.loc[lo:hi]
        if not sl.empty:
            out[label] = {
                "max_bps": float(sl.max()),
                "argmax": sl.idxmax().date().isoformat(),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = TargetsConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrds_targets = [t for t in cfg.targets if t.source == "wrds"]
    if args.target:
        wrds_targets = [t for t in wrds_targets if t.name == args.target]
        if not wrds_targets:
            log.error("target %s not found among WRDS targets in %s",
                      args.target, args.config)
            return 1

    log.info(
        "pulling %d OAS target(s) [source=%s, window=%s..%s]",
        len(wrds_targets), args.source, cfg.window.start, cfg.window.end,
    )
    manifest_entries: list[dict] = []
    n_err = 0
    for tgt in wrds_targets:
        # The OAS module uses "HY"/"IG" short names internally.
        short = tgt.name.replace("_OAS", "")
        try:
            series = fetch_oas(
                target=short,  # type: ignore[arg-type]
                start=cfg.window.start,
                end=cfg.window.end,
                source=args.source,
                cache_root=out_dir,
                use_cache=not args.no_cache,
                wrds_library=args.wrds_library,
                wrds_table=args.wrds_table,
            )
        except Exception as exc:  # noqa: BLE001 — WRDS / FRED raise ad-hoc
            log.error("%s pull failed: %s", tgt.name, exc)
            manifest_entries.append({
                "name": tgt.name, "ticker": tgt.ticker, "source": args.source,
                "status": "error", "error": str(exc),
            })
            n_err += 1
            continue

        entry = {
            "name": tgt.name,
            "ticker": tgt.ticker,
            "source": args.source,
            "rows": int(len(series)),
            "first_date": series.index.min().date().isoformat(),
            "last_date": series.index.max().date().isoformat(),
            "median_bps": float(series.median()),
            "max_bps": float(series.max()),
            "status": "ok",
        }
        if args.sanity_check:
            entry["crisis_peaks"] = _sanity_check(series)
            for label, info in entry["crisis_peaks"].items():
                log.info("  %s peak: %.1f bps on %s", label, info["max_bps"], info["argmax"])
        manifest_entries.append(entry)
        log.info(
            "%s: %d daily bars, %s..%s, median=%.1f bps, max=%.1f bps",
            tgt.name, len(series),
            series.index.min().date(), series.index.max().date(),
            series.median(), series.max(),
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "source": args.source,
        "window": {"start": cfg.window.start.isoformat(), "end": cfg.window.end.isoformat()},
        "entries": manifest_entries,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest to %s/_manifest.json (%d errors)", out_dir, n_err)
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
