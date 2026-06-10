"""Publish v3 Trends Risk Index — weekly + daily cadence.

Inputs:
- ``data/processed/posterior/{HY,IG}_bsts_v3.pkl`` — written by
  ``horizon_sweep_v3.py --mode refit_sweep --save-final-posterior``.
- ``data/raw/`` — weekly Trends cache (v1 layout).
- ``data/raw_daily/`` — daily Trends cache (v3 Phase B layout).

Outputs:
- ``data/processed/risk_index_v3/{HY,IG}_trends_risk_index_weekly.parquet``
- ``data/processed/risk_index_v3/{HY,IG}_trends_risk_index_daily.parquet``
- ``data/processed/risk_index_v3/_metadata.json``

Design call for the daily cadence: we apply the **weekly-trained** posterior
weights (``P(γ_j=1) · β̄_j``) to the **daily-processed** X matrix. The
rolling 5-year z-score normalizes the cross-cadence regime shift. Rebuilding
BSTS on daily-resampled ETF closes is out of scope for v3.

Usage
-----
    PYTHONPATH=src python3 scripts/publish_risk_index_v3.py
    PYTHONPATH=src python3 scripts/publish_risk_index_v3.py --skip-daily   # F.1 only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    p = Path.cwd().resolve()
    while not (p / "src" / "gtrends_bayes").exists():
        if p == p.parent:
            raise RuntimeError("could not find src/gtrends_bayes/ above CWD")
        p = p.parent
    sys.path.insert(0, str(p / "src"))
    return p


def _build_X(cadence: str, cache_root: str | None = None):
    """Build the processed X matrix at the given cadence."""
    from gtrends_bayes.config import PredictorsConfig
    from gtrends_bayes.data.loader import load_predictor_samples, predictor_classes
    from gtrends_bayes.features.library import drop_low_quality_columns
    from gtrends_bayes.preprocessing.pipeline import Pipeline

    pred_cfg = PredictorsConfig.from_yaml("config/predictors.yaml")
    if cache_root is not None:
        long_df = load_predictor_samples(pred_cfg, rename_to_human=True,
                                         cache_root=Path(cache_root))
    else:
        long_df = load_predictor_samples(pred_cfg, rename_to_human=True)
    classes = predictor_classes(pred_cfg, rename_to_human=True)
    pipe = Pipeline(classes=classes, hp_lambda=129_600,
                    weighted_neighbor=True, cadence=cadence)
    processed = pipe.fit_transform(long_df)
    return drop_low_quality_columns(processed, nan_threshold=0.5)


def main() -> int:
    parser = argparse.ArgumentParser(prog="publish_risk_index_v3")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--posterior-dir", default="data/processed/posterior")
    parser.add_argument("--posterior-suffix", default="_bsts_v3.pkl")
    parser.add_argument("--out-dir", default="data/processed/risk_index_v3")
    parser.add_argument("--weekly-cache", default="data/raw")
    parser.add_argument("--daily-cache", default="data/raw_daily")
    parser.add_argument("--skip-daily", action="store_true",
                        help="Build only the weekly Risk Index (F.1).")
    args = parser.parse_args()

    project_root()
    from gtrends_bayes.features.trends_risk_index import build_risk_index

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== building weekly X matrix ===")
    X_weekly = _build_X(cadence="weekly", cache_root=args.weekly_cache)
    print(f"  X_weekly: {X_weekly.shape}, "
          f"{X_weekly.index.min().date()}..{X_weekly.index.max().date()}")

    X_daily = None
    if not args.skip_daily:
        print("\n=== building daily X matrix ===")
        try:
            X_daily = _build_X(cadence="daily", cache_root=args.daily_cache)
            print(f"  X_daily: {X_daily.shape}, "
                  f"{X_daily.index.min().date()}..{X_daily.index.max().date()}")
            # Sanity gate: fall back if too sparse.
            nan_frac = X_daily.isna().mean().mean()
            if nan_frac > 0.30:
                print(f"  WARNING: daily X has {nan_frac:.1%} NaNs — falling back "
                      f"to weekly-only (Decision gate F.2)")
                X_daily = None
        except Exception as exc:
            print(f"  daily X build failed: {exc} — skipping daily Risk Index")
            X_daily = None

    metadata_by_target: dict[str, dict] = {}
    for target in args.targets:
        pkl_path = Path(args.posterior_dir) / f"{target}{args.posterior_suffix}"
        if not pkl_path.exists():
            print(f"\n[{target}] posterior pickle not found at {pkl_path} — skipping")
            continue
        posterior = pickle.loads(pkl_path.read_bytes())
        print(f"\n[{target}] loaded posterior from {pkl_path} "
              f"(X_columns={len(posterior['X_columns'])})")

        tgt_meta: dict = {}

        # Weekly Risk Index
        ri_w = build_risk_index(posterior, X_weekly, target_kind="price",
                                cadence="weekly")
        out_w = out_dir / f"{target}_trends_risk_index_weekly.parquet"
        ri_w.to_parquet(out_w, engine="pyarrow", compression="snappy")
        z = ri_w["zscore_5y"].dropna()
        print(f"  wrote {out_w} ({len(ri_w)} rows; z ∈ "
              f"[{z.min():.2f}, {z.max():.2f}])")
        tgt_meta["weekly"] = {
            "rows": int(len(ri_w)),
            "first_date": ri_w.index.min().date().isoformat(),
            "last_date": ri_w.index.max().date().isoformat(),
            "non_nan_zscore_rows": int(ri_w["zscore_5y"].notna().sum()),
            "tier_counts": {str(k): int(v)
                            for k, v in ri_w["tier"].value_counts().items()},
        }

        # Daily Risk Index (optional)
        if X_daily is not None:
            ri_d = build_risk_index(posterior, X_daily, target_kind="price",
                                    cadence="daily")
            out_d = out_dir / f"{target}_trends_risk_index_daily.parquet"
            ri_d.to_parquet(out_d, engine="pyarrow", compression="snappy")
            z = ri_d["zscore_5y"].dropna()
            print(f"  wrote {out_d} ({len(ri_d)} rows; z ∈ "
                  f"[{z.min():.2f}, {z.max():.2f}])")
            tgt_meta["daily"] = {
                "rows": int(len(ri_d)),
                "first_date": ri_d.index.min().date().isoformat(),
                "last_date": ri_d.index.max().date().isoformat(),
                "non_nan_zscore_rows": int(ri_d["zscore_5y"].notna().sum()),
                "tier_counts": {str(k): int(v)
                                for k, v in ri_d["tier"].value_counts().items()},
            }

        # Top-5 predictors by inclusion prob (sortable)
        summary = posterior["coefficient_summary"]
        if "inclusion_prob" in summary.columns:
            top5 = summary.sort_values("inclusion_prob", ascending=False).head(5)
            tgt_meta["top5_predictors"] = [
                {
                    "predictor": pred,
                    "inclusion_prob": float(row["inclusion_prob"]),
                    "mean_when_included": float(row.get("mean_when_included", 0)),
                }
                for pred, row in top5.iterrows()
            ]

        metadata_by_target[target] = tgt_meta

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v3_posteriors": {
            "source_suffix": args.posterior_suffix,
            "source_dir": str(args.posterior_dir),
        },
        "weighting": "inclusion (P(γ=1) · β̄_when_included · X)",
        "target_kind": "price",
        "weekly": {
            "rolling_window_periods": 260,
            "min_periods_for_zscore": 52,
            "cache_root": args.weekly_cache,
        },
        "daily": {
            "rolling_window_periods": 1260,
            "min_periods_for_zscore": 252,
            "cache_root": args.daily_cache,
            "daily_index_caveat": (
                "Posterior weights P(γ_j=1) · β̄_j come from the weekly BSTS "
                "fit on weekly ETF returns. The daily X matrix is processed "
                "with cadence='daily' (HP λ=14400, YoY periods=252). The "
                "rolling 5y daily z-score normalizes cross-cadence regime "
                "drift. We do NOT refit BSTS on daily-resampled ETF closes — "
                "that is out of scope for v3."
            ),
        },
        "tier_thresholds": [-1.0, 1.0],
        "by_target": metadata_by_target,
    }
    meta_path = out_dir / "_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
