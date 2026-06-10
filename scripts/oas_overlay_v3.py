"""ETF-vs-OAS proxy correlation analysis (v3 reference overlay).

Reads ``data/csv/BAML{H0A0HYM2,C0A0CM}.csv`` (FRED daily OAS, 2023-05+),
resamples to weekly Sunday-anchor to match the ETF weekly cache, and
computes the empirical correlation between ETF log-returns and ΔOAS over
the 2023-2026 overlap window. Output:

- ``data/processed/oas_overlay/correlation.json`` — proxy quality numbers.
- Optionally prints a markdown-ready table for Notebook 13.

This is the v3 "reference overlay" deliverable from the plan's OAS-CSVs
note. It does NOT retrain BSTS on OAS (window too short — see plan).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    p = Path.cwd().resolve()
    while not (p / "src" / "gtrends_bayes").exists():
        if p == p.parent:
            raise RuntimeError("could not find src/gtrends_bayes/ above CWD")
        p = p.parent
    sys.path.insert(0, str(p / "src"))
    return p


# FRED units are percent; ×100 to get basis points (PM convention).
PCT_TO_BPS = 100.0

OAS_CSVS = {
    "HY": ("data/csv/BAMLH0A0HYM2.csv", "BAMLH0A0HYM2"),
    "IG": ("data/csv/BAMLC0A0CM.csv",   "BAMLC0A0CM"),
}


def load_oas(target: str, anchor: str = "SUN") -> tuple[pd.Series, pd.Series]:
    """Return (daily OAS bps, weekly OAS bps Sunday-anchored)."""
    path, col = OAS_CSVS[target]
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    daily = (df.set_index("observation_date")[col]
               .astype(float) * PCT_TO_BPS).rename(f"{target}_OAS_bps")
    # Sunday-anchored weekly bars to match the existing ETF target cache.
    weekly = daily.resample(f"W-{anchor.upper()}").last().dropna()
    return daily.dropna(), weekly


def compute_correlation(target: str, etf_weekly: pd.Series,
                        oas_weekly: pd.Series) -> dict:
    """Compute correlation between ETF log-returns and ΔOAS over overlap."""
    etf_dlog = np.log(etf_weekly).diff().rename("etf_dlog")
    oas_diff = oas_weekly.diff().rename("oas_diff_bps")
    df = pd.concat([etf_dlog, oas_diff], axis=1).dropna()
    if len(df) < 10:
        return {"n_obs": int(len(df)), "pearson": None, "spearman": None}
    return {
        "target": target,
        "n_obs": int(len(df)),
        "overlap_start": df.index.min().date().isoformat(),
        "overlap_end":   df.index.max().date().isoformat(),
        "pearson":  float(df["etf_dlog"].corr(df["oas_diff_bps"], method="pearson")),
        "spearman": float(df["etf_dlog"].corr(df["oas_diff_bps"], method="spearman")),
        "etf_dlog_std": float(df["etf_dlog"].std()),
        "oas_diff_std_bps": float(df["oas_diff_bps"].std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="oas_overlay_v3")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--etf-targets-dir", default="data/raw/targets")
    parser.add_argument("--out-dir", default="data/processed/oas_overlay")
    parser.add_argument(
        "--write-targets", action="store_true",
        help="Also write weekly OAS bps to data/raw/targets/{TARGET}_OAS.parquet "
             "so the auxiliary OAS-direct BSTS (v5.1 sub-model) can load these "
             "via the standard load_target() path. The column inside each parquet "
             "is named '{TARGET}_OAS' to match config/targets.yaml.",
    )
    parser.add_argument(
        "--targets-out-dir", default="data/raw/targets",
        help="Where --write-targets puts the OAS target parquets.",
    )
    args = parser.parse_args()

    project_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    print("=== Loading FRED OAS CSVs ===")
    for target in args.targets:
        daily_oas, weekly_oas = load_oas(target)
        print(f"{target}: {len(daily_oas)} daily, {len(weekly_oas)} weekly  "
              f"({daily_oas.index.min().date()} -> {daily_oas.index.max().date()})  "
              f"latest = {daily_oas.iloc[-1]:.1f} bps")
        # Cache the daily + weekly OAS in parquet for downstream re-use.
        out_dir.joinpath(f"{target}_OAS_daily.parquet").write_bytes(b"")  # placeholder
        daily_oas.to_frame().to_parquet(out_dir / f"{target}_OAS_daily.parquet",
                                        engine="pyarrow", compression="snappy")
        weekly_oas.to_frame().to_parquet(out_dir / f"{target}_OAS_weekly.parquet",
                                         engine="pyarrow", compression="snappy")

        # If requested, also emit a target-style parquet that load_target()
        # can read directly. The auxiliary OAS-direct BSTS pipeline expects
        # `data/raw/targets/{TARGET}_OAS.parquet` with a single column named
        # `{TARGET}_OAS` and a 'Date'-named index (matches the ETF parquets).
        if args.write_targets:
            tgt_path = Path(args.targets_out_dir) / f"{target}_OAS.parquet"
            tgt_path.parent.mkdir(parents=True, exist_ok=True)
            series = weekly_oas.copy()
            series.index.name = "Date"
            series.name = f"{target}_OAS"
            series.to_frame().to_parquet(tgt_path, engine="pyarrow",
                                         compression="snappy")
            print(f"  wrote {tgt_path} ({len(series)} weekly bars)")

        # ETF cache (Sunday-anchored weekly)
        etf_path = Path(args.etf_targets_dir) / f"{target}.parquet"
        if not etf_path.exists():
            print(f"  WARN: ETF cache {etf_path} not found; skipping correlation")
            results[target] = {"oas_only": True}
            continue
        etf_weekly = pd.read_parquet(etf_path).iloc[:, 0]
        etf_weekly.index = pd.DatetimeIndex(etf_weekly.index)

        corr = compute_correlation(target, etf_weekly, weekly_oas)
        results[target] = corr
        print(f"  ETF dlog vs ΔOAS  (n={corr['n_obs']}):  "
              f"Pearson={corr['pearson']:+.3f}, Spearman={corr['spearman']:+.3f}")
        print(f"    overlap {corr['overlap_start']} -> {corr['overlap_end']}, "
              f"etf σ(dlog)={corr['etf_dlog_std']:.4f}, "
              f"OAS σ(Δbps)={corr['oas_diff_std_bps']:.2f}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "FRED via data/csv/BAML*.csv (user-supplied 2026-05-18)",
        "units": "OAS in basis points (×100 from FRED's percent)",
        "weekly_anchor": "SUN",
        "by_target": results,
        "interpretation": (
            "Correlation sign should be NEGATIVE: when OAS widens (Δ > 0), "
            "the corresponding ETF price falls (log-return < 0). Magnitude "
            "in the [-0.7, -0.9] range indicates a high-quality proxy. "
            "Weakness in magnitude flags regime-specific deviations the PM "
            "should be aware of."
        ),
    }
    out_path = out_dir / "correlation.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # ---- Markdown table for Notebook 13 ----
    print("\n=== ETF-vs-OAS proxy quality (markdown) ===")
    print("| Target | n_weeks | Pearson | Spearman | overlap |")
    print("|---|---:|---:|---:|---|")
    for tgt, r in results.items():
        if r.get("oas_only"):
            continue
        print(f"| {tgt} | {r['n_obs']} | {r['pearson']:+.3f} | "
              f"{r['spearman']:+.3f} | {r['overlap_start']} → {r['overlap_end']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
