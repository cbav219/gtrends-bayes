"""Cross-check daily-cadence Trends pulls against the existing weekly cache.

For each of the 41 predictors:
  1. Load weekly samples from ``data/raw/`` (existing v1/v2 cache).
  2. Load daily samples from ``data/raw_daily/`` (Phase B pull).
  3. Aggregate the daily series up to weekly (mean within ISO week).
  4. Compare ``weekly_from_daily`` vs directly-pulled ``weekly``.
  5. Also report cross-sample std at daily resolution per query — drop
     queries whose std > 30 across the 6 samples (too noisy after stitching).

Outputs ``data/processed/validation/daily_vs_weekly.json`` and prints a
summary table. v3 gates Phases C–F on the top-10 inclusion-probability
predictors agreeing within 5% mean abs error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gtrends_bayes.config import PredictorsConfig
from gtrends_bayes.data.cache import cache_path, read_sample, slugify
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


def _load_stitched_samples(slug: str, geo: str, start, end, n_samples: int,
                           root: Path) -> pd.DataFrame:
    """Concatenate cached samples for one predictor; returns long-form."""
    frames = []
    for i in range(n_samples):
        p = cache_path(slug, geo, start, end, i, root=root)
        if not p.exists():
            continue
        frames.append(read_sample(p))
    if not frames:
        return pd.DataFrame(columns=["date", "query", "sample_idx", "svi"])
    return pd.concat(frames, ignore_index=True)


def _daily_to_weekly(df: pd.DataFrame, anchor: str = "SUN") -> pd.DataFrame:
    """Aggregate daily long-form into weekly Sunday-aligned bars (per sample)."""
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rule = f"W-{anchor.upper()}"
    out = (
        df.set_index("date")
          .groupby(["query", "sample_idx"], group_keys=False)["svi"]
          .resample(rule)
          .mean()
          .reset_index()
    )
    return out


def _per_query_mape(weekly_direct: pd.DataFrame,
                    weekly_from_daily: pd.DataFrame) -> float:
    """Mean absolute percentage error of per-week mean SVI between the two."""
    # Average over samples → one series per query × date.
    direct = weekly_direct.groupby("date")["svi"].mean()
    from_daily = weekly_from_daily.groupby("date")["svi"].mean()
    aligned = direct.align(from_daily, join="inner")
    if len(aligned[0]) == 0:
        return float("nan")
    eps = 1e-6
    return float(((aligned[0] - aligned[1]).abs() / (aligned[0].abs() + eps)).mean())


def _cross_sample_std(df: pd.DataFrame) -> float:
    """Mean across dates of cross-sample std (one value per query)."""
    if df.empty:
        return float("nan")
    pivot = df.pivot_table(index="date", columns="sample_idx", values="svi")
    return float(pivot.std(axis=1).mean())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="validate_daily_vs_weekly")
    p.add_argument("--config", default="config/predictors.yaml")
    p.add_argument("--weekly-root", default="data/raw")
    p.add_argument("--daily-root", default="data/raw_daily")
    p.add_argument("--out", default="data/processed/validation/daily_vs_weekly.json")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--mape-threshold", type=float, default=0.05,
                   help="Max acceptable per-predictor MAPE (default 5%).")
    p.add_argument("--std-threshold", type=float, default=30.0,
                   help="Cross-sample std (daily) threshold; queries above "
                        "this are flagged as too-noisy-to-use.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PredictorsConfig.from_yaml(args.config)
    n_samples = args.n_samples or cfg.sampling.n_samples
    weekly_root = Path(args.weekly_root)
    daily_root = Path(args.daily_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "validating %d predictors (weekly=%s, daily=%s, window=%s..%s)",
        len(cfg.predictors), weekly_root, daily_root,
        cfg.window.start, cfg.window.end,
    )

    rows = []
    for pred in cfg.predictors:
        ident = pred.id if pred.kind == "category" else pred.mid
        slug = slugify(ident)
        weekly = _load_stitched_samples(
            slug, cfg.geo, cfg.window.start, cfg.window.end, n_samples,
            root=weekly_root,
        )
        daily = _load_stitched_samples(
            slug, cfg.geo, cfg.window.start, cfg.window.end, n_samples,
            root=daily_root,
        )
        if weekly.empty or daily.empty:
            rows.append({
                "name": pred.name, "kind": pred.kind, "group": pred.group,
                "status": "missing",
                "weekly_n": len(weekly), "daily_n": len(daily),
            })
            continue

        weekly_from_daily = _daily_to_weekly(daily)
        mape = _per_query_mape(weekly, weekly_from_daily)
        daily_std = _cross_sample_std(daily)
        rows.append({
            "name": pred.name, "kind": pred.kind, "group": pred.group,
            "status": "ok",
            "mape_weekly_vs_from_daily": mape,
            "daily_cross_sample_std": daily_std,
            "passes_mape_threshold": bool(np.isfinite(mape) and mape <= args.mape_threshold),
            "passes_std_threshold": bool(np.isfinite(daily_std) and daily_std <= args.std_threshold),
        })

    summary = {
        "config": str(Path(args.config).resolve()),
        "window": {"start": cfg.window.start.isoformat(), "end": cfg.window.end.isoformat()},
        "n_samples": n_samples,
        "mape_threshold": args.mape_threshold,
        "std_threshold": args.std_threshold,
        "n_predictors": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "ok"),
        "n_missing": sum(1 for r in rows if r["status"] == "missing"),
        "n_passing_mape": sum(1 for r in rows if r.get("passes_mape_threshold")),
        "n_passing_std": sum(1 for r in rows if r.get("passes_std_threshold")),
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "validation: %d/%d predictors pass MAPE ≤ %.0f%%; "
        "%d/%d pass daily-std ≤ %.0f",
        summary["n_passing_mape"], summary["n_ok"], 100 * args.mape_threshold,
        summary["n_passing_std"], summary["n_ok"], args.std_threshold,
    )
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
