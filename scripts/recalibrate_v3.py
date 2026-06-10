"""Conformal recalibration on v3 raw walk-forward parquets (D.2).

Drop-in v3 analogue of ``recalibrate_coverage.py``: reuses
``backtest.recalibrate.fit_per_level`` but reads from the v3 cache layout
``data/processed/backtest/raw_v3/{target}_{slug}_re{re}_ar{ar}.parquet``.

Writes ``data/processed/backtest/recalibration_alphas_v3.json`` and prints
the v2.1 acceptance gate (in-sample cov_80_recal ∈ [0.75, 0.85] for ≥ 7/8
cells).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def project_root() -> Path:
    p = Path.cwd().resolve()
    while not (p / "src" / "gtrends_bayes").exists():
        if p == p.parent:
            raise RuntimeError("could not find src/gtrends_bayes/ above CWD")
        p = p.parent
    sys.path.insert(0, str(p / "src"))
    return p


# v3 model set + their (ar_p, ar_aware) tuples for raw_v3 file naming.
V3_MODELS: list[tuple[str, str, int]] = [
    # (model_name, slug, ar_for_path)
    ("BSTS (Trends)",   "BSTS_Trends",     0),
    ("StackedResidual", "StackedResidual", 4),  # locked v3 default
    ("AR(p)",           "ARp",             4),
    ("Naive RW",        "Naive_RW",        0),
]


def main() -> int:
    parser = argparse.ArgumentParser(prog="recalibrate_v3")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--refit-every", type=int, default=4)
    parser.add_argument("--raw-dir", default="data/processed/backtest/raw_v3")
    parser.add_argument("--alpha-out",
                        default="data/processed/backtest/recalibration_alphas_v3.json")
    parser.add_argument("--csv-update",
                        default="data/processed/backtest/comparison_table_v3.csv")
    parser.add_argument("--val-split", type=float, default=0.5,
                        help="OOS robustness split (0 to skip OOS check).")
    args = parser.parse_args()

    project_root()
    from gtrends_bayes.backtest.recalibrate import fit_per_level
    from gtrends_bayes.config import TargetsConfig
    from gtrends_bayes.features.library import load_target

    raw_dir = Path(args.raw_dir)
    val_split = args.val_split if args.val_split > 0 else None

    tgt_cfg = TargetsConfig.from_yaml("config/targets.yaml")
    alpha_results: dict[str, dict[str, dict]] = {}
    recal_rows: list[dict] = []

    for target in args.targets:
        y = load_target(target, tgt_cfg)
        alpha_results.setdefault(target, {})
        print(f"\n=== {target} ===")

        for model_name, slug, ar_p in V3_MODELS:
            path = raw_dir / f"{target}_{slug}_re{args.refit_every}_ar{ar_p}.parquet"
            if not path.exists():
                print(f"  [{model_name}] missing parquet at {path} — skipping")
                continue
            df = pd.read_parquet(path).copy()
            df.index = pd.DatetimeIndex(df.index)
            common = df.index.intersection(y.index)
            y_aligned = y.loc[common]
            bands = df.loc[common]

            per_level = fit_per_level(
                y_aligned, bands, levels=(0.50, 0.80, 0.95),
                val_split=val_split, median_col="q500",
            )
            alpha_results[target][model_name] = per_level
            row = {
                "target": target, "model": model_name,
                "alpha_50":  round(per_level[0.50]["alpha"], 4),
                "alpha_80":  round(per_level[0.80]["alpha"], 4),
                "alpha_95":  round(per_level[0.95]["alpha"], 4),
                "cov_50_pre":   round(per_level[0.50]["empirical_pre_full"], 4),
                "cov_80_pre":   round(per_level[0.80]["empirical_pre_full"], 4),
                "cov_95_pre":   round(per_level[0.95]["empirical_pre_full"], 4),
                "cov_50_recal": round(per_level[0.50]["empirical_post_full"], 4),
                "cov_80_recal": round(per_level[0.80]["empirical_post_full"], 4),
                "cov_95_recal": round(per_level[0.95]["empirical_post_full"], 4),
            }
            if val_split is not None:
                row.update({
                    "alpha_80_oos":     round(per_level[0.80]["alpha_oos"], 4),
                    "cov_80_oos_pre":   round(per_level[0.80]["empirical_pre_test"], 4),
                    "cov_80_oos_post":  round(per_level[0.80]["empirical_post_test"], 4),
                })
            recal_rows.append(row)
            print(f"  {model_name:>16s} | α_80={row['alpha_80']:.2f}"
                  f" | cov_80: {row['cov_80_pre']:.2f} → {row['cov_80_recal']:.2f}"
                  + (f"  |  OOS α_80={row['alpha_80_oos']:.2f}, "
                     f"cov={row['cov_80_oos_post']:.2f}" if val_split is not None else ""))

    Path(args.alpha_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.alpha_out, "w") as f:
        json.dump(alpha_results, f, indent=2, default=str)
    print(f"\nwrote {args.alpha_out}")

    recal_df = pd.DataFrame(recal_rows)
    csv_path = Path(args.csv_update)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        base = pd.read_csv(csv_path)
        recal_col_prefixes = ("alpha_", "cov_50_pre", "cov_80_pre", "cov_95_pre",
                              "cov_50_recal", "cov_80_recal", "cov_95_recal",
                              "cov_80_oos", "n_val", "n_test")
        keep_cols = [c for c in base.columns if not c.startswith(recal_col_prefixes)]
        base = base[keep_cols]
        merged = base.merge(recal_df, on=["target", "model"], how="left")
        merged.to_csv(csv_path, index=False)
        print(f"appended recal columns to {csv_path}")
    else:
        recal_df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")

    print("\n=== v2.1 D.3 acceptance: in-sample cov_80_recal ∈ [0.75, 0.85] ===")
    pass_count = 0
    for r in recal_rows:
        v = r["cov_80_recal"]
        ok = 0.75 <= v <= 0.85
        pass_count += int(ok)
        mark = "PASS" if ok else "FAIL"
        extra = ""
        if val_split is not None:
            extra = f"  |  OOS cov_80 post-α = {r['cov_80_oos_post']:.2f}"
        print(f"  [{mark}] {r['target']:>3s} | {r['model']:>16s} | "
              f"pre={r['cov_80_pre']:.2f} → recal={v:.2f}{extra}")
    print(f"\n{pass_count}/{len(recal_rows)} cells in band (in-sample).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
