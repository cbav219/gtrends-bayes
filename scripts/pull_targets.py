"""CLI: fetch financial targets + macro controls from FRED, cache to data/raw/targets/.

Targets and controls are listed in ``config/targets.yaml``. Each FRED-sourced
series is fetched daily, resampled to weekly Sunday-aligned bars to match the
Trends weekly cadence, and written as a single-column Parquet at
``data/raw/targets/<name>.parquet``.

"Derived" controls (e.g. the 2y10y slope) are NOT computed here — they live in
``features.library``. This script also pulls every FRED ticker referenced in a
"derived" formula so the feature layer has its dependencies available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from gtrends_bayes.config import TargetsConfig
from gtrends_bayes.data.financial import fetch_target, resample_weekly
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)

DEFAULT_TARGETS_ROOT = Path("data/raw/targets")

# FRED tickers that appear in "derived" formulas — pulled as-is so feature code
# can reference them by ticker. Keys are lower-cased ticker, values are the
# parquet basename to write.
_FORMULA_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9]+)\b")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pull_targets",
        description="Fetch FRED targets + macro controls into data/raw/targets/.",
    )
    p.add_argument("--config", default="config/targets.yaml")
    p.add_argument("--out-dir", default=str(DEFAULT_TARGETS_ROOT))
    p.add_argument("--week-anchor", default="SUN", help="weekly resampling anchor")
    return p


def _formula_dependencies(formula: str) -> list[str]:
    """Extract bare FRED-ticker tokens from a derived formula string."""
    return [t for t in _FORMULA_TICKER_RE.findall(formula) if t.isupper() and len(t) > 1]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = TargetsConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan one fetch per unique (ticker, source) pair. FRED tickers referenced
    # in a "derived" control formula are added so the feature layer can compose
    # them. The friendly name becomes the parquet filename.
    plan: dict[tuple[str, str], str] = {}   # (ticker, source) -> friendly name
    for t in cfg.targets:
        if t.source in ("fred", "yfinance"):
            plan[(t.ticker, t.source)] = t.name
    for c in cfg.controls:
        if c.source in ("fred", "yfinance") and c.ticker:
            plan[(c.ticker, c.source)] = c.name
        elif c.source == "derived" and c.formula:
            for tkr in _formula_dependencies(c.formula):
                plan.setdefault((tkr, "fred"), tkr.lower())

    n_fred = sum(1 for (_t, s) in plan if s == "fred")
    n_yf = sum(1 for (_t, s) in plan if s == "yfinance")
    log.info(
        "fetching %d series (%d FRED, %d yfinance) (window=%s..%s, anchor=W-%s)",
        len(plan), n_fred, n_yf, cfg.window.start, cfg.window.end, args.week_anchor,
    )

    manifest_entries: list[dict] = []
    for (ticker, source), friendly in plan.items():
        try:
            daily = fetch_target(friendly, source, ticker, cfg.window.start, cfg.window.end)
        except Exception as exc:  # noqa: BLE001
            log.error("%s fetch failed for %s (%s): %s", source, friendly, ticker, exc)
            manifest_entries.append({
                "name": friendly, "ticker": ticker, "source": source,
                "status": "error", "error": str(exc),
            })
            continue
        weekly = resample_weekly(daily, anchor=args.week_anchor).rename(friendly)
        path = out_dir / f"{friendly}.parquet"
        weekly.to_frame().to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(
            "wrote %s (%s) -> %s (%d weekly bars, %s..%s)",
            ticker, source, path, len(weekly),
            weekly.index.min().date(), weekly.index.max().date(),
        )
        manifest_entries.append({
            "name": friendly,
            "ticker": ticker,
            "source": source,
            "rows_weekly": len(weekly),
            "first_date": weekly.index.min().date().isoformat(),
            "last_date": weekly.index.max().date().isoformat(),
            "status": "ok",
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "window": {"start": cfg.window.start.isoformat(), "end": cfg.window.end.isoformat()},
        "week_anchor": args.week_anchor,
        "entries": manifest_entries,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest to %s/_manifest.json", out_dir)
    n_err = sum(1 for e in manifest_entries if e["status"] != "ok")
    log.info("done: %d entries (%d errors)", len(manifest_entries), n_err)
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
