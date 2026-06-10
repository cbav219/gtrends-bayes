"""Evaluate the v3 Trends Risk Index at weekly + daily cadence.

Inputs:
- ``data/processed/risk_index_v3/{HY,IG}_trends_risk_index_{weekly,daily}.parquet``
- ``data/raw/targets/{HY,IG}.parquet`` + ``data/raw/targets/vix.parquet`` (weekly bars).
- Daily VIX + daily HYG/LQD: fetched on demand via yfinance / FRED and
  cached at ``data/raw/targets/{HY,IG,vix}_daily.parquet``.

Four evaluation tests per (cadence, target):
1. Granger vs VIX (with VIX as control).
2. Quantile portfolios (5 buckets; monotone mean forward Δlog target).
3. Crisis recall (top-decile flag in 4 weeks before COVID/gilt/SVB).
4. Lead/lag vs VIX (cross-correlation; index leading VIX is the win).

Output: ``data/processed/risk_index_v3/_evaluation.json`` with
``{weekly: {HY, IG}, daily: {HY, IG}}`` structure.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def project_root() -> Path:
    p = Path.cwd().resolve()
    while not (p / "src" / "gtrends_bayes").exists():
        if p == p.parent:
            raise RuntimeError("could not find src/gtrends_bayes/ above CWD")
        p = p.parent
    sys.path.insert(0, str(p / "src"))
    sys.path.insert(0, str(p / "scripts"))
    return p


def _fetch_daily_if_missing(name: str, source: str, ticker: str,
                            start: date, end: date,
                            cache_dir: Path) -> pd.Series:
    """Fetch a daily series via fetch_target; cache for next time."""
    from dotenv import load_dotenv
    from gtrends_bayes.data.financial import fetch_target

    load_dotenv()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_daily.parquet"
    if path.exists():
        s = pd.read_parquet(path).iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index)
        return s
    print(f"  fetching {name} ({source}:{ticker}) daily {start}..{end}")
    s = fetch_target(name, source, ticker, start, end).rename(name)
    s.index = pd.DatetimeIndex(s.index)
    s.to_frame().to_parquet(path, engine="pyarrow", compression="snappy")
    print(f"    cached -> {path} ({len(s)} rows)")
    return s


def evaluate_one(target: str, cadence: str,
                 z: pd.Series, y_levels: pd.Series, vix_levels: pd.Series,
                 max_corr_lag: int, granger_lag: int) -> dict:
    """Run the four tests for one (cadence, target) cell."""
    from evaluate_risk_index import (
        granger_test, quantile_portfolios, crisis_recall, cross_correlation,
    )
    from gtrends_bayes.features.trends_risk_index import crisis_windows

    crises = crisis_windows()

    y_dlog = np.log(y_levels).diff().rename(f"{target}_dlog")
    vix_dlog = np.log(vix_levels).diff().rename("vix_dlog")

    # Align cadences: if daily, z + y + vix should all be daily; if weekly, all weekly.
    # The risk_index parquet is already at the right cadence; y / vix are at the
    # cadence specified by the caller. We align by inner-joining indices.
    y_fwd = y_dlog.shift(-1)

    granger = granger_test(y_dlog, z, vix_dlog, max_lag=granger_lag)
    quants = quantile_portfolios(z, y_fwd, n_quantiles=5)
    crecall = crisis_recall(z, crises)
    cc = cross_correlation(z, vix_dlog, max_k=max_corr_lag)

    print(f"  Granger F={granger['f_stat']:.2f}, p={granger['p_value']:.4f}, "
          f"ΔR²={granger['delta_r2']:.4f}")
    print(f"  Quantile spread (top - bottom): "
          f"{quants['spread_top_minus_bottom']:+.5f} "
          f"(monotone={quants['monotone']})")
    print(f"  Crisis recall: {crecall['recall']:.0%}")
    print(f"  Lead/lag vs VIX: best k={cc['best_lag']}, "
          f"corr={cc['best_corr']:.3f}  ({cc['interpretation']})")

    return {
        "n_obs": int(z.dropna().shape[0]),
        "zscore_min": float(z.dropna().min()) if z.dropna().shape[0] else None,
        "zscore_max": float(z.dropna().max()) if z.dropna().shape[0] else None,
        "granger": granger,
        "quantile_portfolios": quants,
        "crisis_recall": crecall,
        "cross_correlation_vs_vix": cc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluate_risk_index_v3")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--risk-index-dir",
                        default="data/processed/risk_index_v3")
    parser.add_argument("--targets-dir", default="data/raw/targets")
    parser.add_argument("--out",
                        default="data/processed/risk_index_v3/_evaluation.json")
    parser.add_argument("--max-granger-lag", type=int, default=4)
    parser.add_argument("--daily-window-start", default="2008-01-01")
    parser.add_argument("--daily-window-end", default="2026-04-30")
    args = parser.parse_args()

    project_root()
    daily_start = date.fromisoformat(args.daily_window_start)
    daily_end = date.fromisoformat(args.daily_window_end)

    targets_dir = Path(args.targets_dir)
    ri_dir = Path(args.risk_index_dir)

    # ---- weekly inputs ----
    vix_weekly_path = targets_dir / "vix.parquet"
    if not vix_weekly_path.exists():
        print(f"ERROR: weekly VIX cache not found at {vix_weekly_path}",
              file=sys.stderr)
        return 1
    vix_weekly = pd.read_parquet(vix_weekly_path).iloc[:, 0]
    vix_weekly.index = pd.DatetimeIndex(vix_weekly.index)

    # ---- daily inputs (fetch on demand) ----
    print("=== ensuring daily targets cache ===")
    vix_daily = _fetch_daily_if_missing(
        "vix", "fred", "VIXCLS", daily_start, daily_end, targets_dir)
    y_daily = {}
    for target, ticker in [("HY", "HYG"), ("IG", "LQD")]:
        y_daily[target] = _fetch_daily_if_missing(
            target, "yfinance", ticker, daily_start, daily_end, targets_dir)

    by_cadence: dict[str, dict] = {"weekly": {}, "daily": {}}

    for target in args.targets:
        y_weekly_path = targets_dir / f"{target}.parquet"
        if not y_weekly_path.exists():
            print(f"skipping {target}: weekly target {y_weekly_path} missing")
            continue
        y_weekly = pd.read_parquet(y_weekly_path).iloc[:, 0]
        y_weekly.index = pd.DatetimeIndex(y_weekly.index)

        # Weekly
        ri_w_path = ri_dir / f"{target}_trends_risk_index_weekly.parquet"
        if ri_w_path.exists():
            ri_w = pd.read_parquet(ri_w_path)
            ri_w.index = pd.DatetimeIndex(ri_w.index)
            print(f"\n=== weekly | {target} ===")
            by_cadence["weekly"][target] = evaluate_one(
                target, "weekly", ri_w["zscore_5y"],
                y_weekly, vix_weekly,
                max_corr_lag=4, granger_lag=args.max_granger_lag,
            )
        else:
            print(f"\nskipping weekly {target}: {ri_w_path} missing")

        # Daily
        ri_d_path = ri_dir / f"{target}_trends_risk_index_daily.parquet"
        if ri_d_path.exists():
            ri_d = pd.read_parquet(ri_d_path)
            ri_d.index = pd.DatetimeIndex(ri_d.index)
            print(f"\n=== daily  | {target} ===")
            by_cadence["daily"][target] = evaluate_one(
                target, "daily", ri_d["zscore_5y"],
                y_daily[target], vix_daily,
                max_corr_lag=20, granger_lag=args.max_granger_lag,
            )
        else:
            print(f"\nskipping daily {target}: {ri_d_path} missing")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "by_cadence": by_cadence,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
