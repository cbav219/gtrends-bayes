"""Run the multi-horizon walk-forward sweep across 4 models × 2 targets.

For each (target, model, horizon), score:
  - RMSE
  - hit_rate (directional, cumulative-change sense)
  - ic_spearman (rank correlation of h-period change forecasts vs realized)
  - brier_score (P(widening) vs realized widening)
  - auc (same predictor, ROC-AUC against realized widening)
  - precision_widening / recall_widening (at the binary threshold P > 0.5)

Output: long-format ``data/processed/backtest/horizon_sweep.csv`` with one row
per (target, model, horizon).

Usage
-----
    python scripts/horizon_sweep.py
    python scripts/horizon_sweep.py --horizons 1 4 13 --niter 600 --burn 60
    python scripts/horizon_sweep.py --targets HY  # one target only
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
    return p


def build_features(target_name: str):
    from gtrends_bayes.config import PredictorsConfig, TargetsConfig
    from gtrends_bayes.data.loader import load_predictor_samples, predictor_classes
    from gtrends_bayes.features.library import (
        add_market_controls,
        build_feature_matrix,
        drop_low_quality_columns,
        load_market_controls,
        load_target,
    )
    from gtrends_bayes.preprocessing.pipeline import Pipeline

    pred_cfg = PredictorsConfig.from_yaml("config/predictors.yaml")
    tgt_cfg = TargetsConfig.from_yaml("config/targets.yaml")
    long_df = load_predictor_samples(pred_cfg, rename_to_human=True)
    classes = predictor_classes(pred_cfg, rename_to_human=True)
    pipe = Pipeline(classes=classes, hp_lambda=129_600, weighted_neighbor=True)
    processed = pipe.fit_transform(long_df)
    processed_clean = drop_low_quality_columns(processed, nan_threshold=0.5)
    target = load_target(target_name, tgt_cfg)
    X, y = build_feature_matrix(processed_clean, target, train_eligible=pipe.train_eligible_)
    controls = load_market_controls(tgt_cfg)
    X, _ = add_market_controls(X, controls)
    X = X.dropna()
    y = y.loc[X.index]
    return X, y


def _load_expected_predictors(target_name: str, fallback: int = 5) -> int:
    """Read v2.1 per-target ``expected_model_size`` from config/model.yaml."""
    import yaml as _yaml
    from pathlib import Path
    p = Path("config/model.yaml")
    if not p.exists():
        return fallback
    raw = _yaml.safe_load(p.read_text()) or {}
    prior = raw.get("bsts", {}).get("prior", {})
    per_target = prior.get("expected_model_size_per_target") or {}
    return int(per_target.get(target_name, prior.get("expected_model_size", fallback)))


def run_walk_forward_multihorizon(X, y, horizons, bsts_niter, bsts_burn,
                                  train_window, refit_every,
                                  expected_predictors: int = 5):
    from gtrends_bayes.backtest.walk_forward import WalkForward
    from gtrends_bayes.models.baseline import AR_p, AR_VIX, NaiveRW
    from gtrends_bayes.models.bsts import BSTS, reset_r_models

    wf = WalkForward(train_window=train_window, step=1, horizons=horizons,
                     refit_every=refit_every, publication_lag=1)
    out: dict[str, pd.DataFrame] = {}
    out["BSTS (Trends)"] = wf.run(
        lambda: BSTS(n_seasons=52, expected_predictors=expected_predictors,
                     niter=bsts_niter, burn=bsts_burn, seed=42),
        X, y, n_draws=400,
    )
    reset_r_models()
    X_empty = X.iloc[:, :0]
    out["AR(4)"] = wf.run(lambda: AR_p(p=4), X_empty, y, n_draws=200)
    out["Naive RW"] = wf.run(NaiveRW, X_empty, y, n_draws=200)
    X_vix = X[["vix"]] if "vix" in X.columns else X_empty
    out["AR(4) + VIX"] = wf.run(lambda: AR_VIX(p=4), X_vix, y, n_draws=200)
    return out


def score_horizon(model_name: str, target: str, horizon: int,
                  results_h: pd.DataFrame, y: pd.Series) -> dict:
    """Compute v2 metrics on a single (target, model, horizon) slice."""
    from gtrends_bayes.backtest.metrics import (
        auc_roc, brier_score, information_coefficient,
        precision_recall_widening, rmse,
    )

    common = results_h.index.intersection(y.index)
    df = results_h.loc[common]
    y_true = df["y_true"]
    y_pred = df["y_pred_mean"]

    # Cumulative h-period prior y (the level just before the forecast period).
    prev_y = y.shift(horizon).reindex(common)

    # Δy over the h-period horizon, predicted vs realized.
    dy_pred = y_pred - prev_y
    dy_true = y_true - prev_y

    # Direction match on cumulative change.
    valid = dy_pred.notna() & dy_true.notna()
    hit_rate = float((np.sign(dy_pred[valid]) == np.sign(dy_true[valid])).mean()) \
        if valid.any() else float("nan")

    # IC on cumulative changes (rank corr).
    ic = information_coefficient(
        pd.Series(dy_pred.values, index=dy_pred.index),
        pd.Series(dy_true.values, index=dy_true.index),
        method="spearman",
    )

    # P(widening) from the posterior bands. Use the share of band quantiles
    # implying widening direction. ETF price target → widening = price drops.
    bands = df[["q025", "q050", "q100", "q250", "q500", "q750", "q900", "q975"]]
    p_widening = (bands.lt(prev_y, axis=0).sum(axis=1) / bands.shape[1])
    y_widening = (dy_true < -0.25).astype(int)  # 0.25 USD ≈ 25¢ ETF move ≈ stress

    brier = brier_score(p_widening, y_widening)
    auc = auc_roc(p_widening, y_widening)

    pr = precision_recall_widening(
        (p_widening > 0.5).astype(int), y_true,
        widening_threshold=0.25, direction="decrease",
    )

    return {
        "target": target,
        "model": model_name,
        "horizon": horizon,
        "n_obs": int(valid.sum()),
        "rmse": round(rmse(y_true, y_pred), 4),
        "hit_rate": round(hit_rate, 4),
        "ic_spearman": round(ic, 4) if not np.isnan(ic) else np.nan,
        "brier_score": round(brier, 4) if not np.isnan(brier) else np.nan,
        "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "precision_widening": (round(pr["precision"], 4)
                               if not np.isnan(pr["precision"]) else np.nan),
        "recall_widening": (round(pr["recall"], 4)
                            if not np.isnan(pr["recall"]) else np.nan),
        "n_widening_events": pr["n_events"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="horizon_sweep")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 13])
    parser.add_argument("--niter", type=int, default=900)
    parser.add_argument("--burn", type=int, default=150)
    parser.add_argument("--train-window", type=int, default=260)
    parser.add_argument("--refit-every", type=int, default=13)
    parser.add_argument("--out", type=str,
                        default="data/processed/backtest/horizon_sweep.csv")
    args = parser.parse_args()

    project_root()
    print(f"horizons: {args.horizons}, BSTS niter={args.niter}, burn={args.burn}")

    rows: list[dict] = []
    for target in args.targets:
        print(f"\n=== {target} ===")
        X, y = build_features(target)
        print(f"X={X.shape}, y={y.shape}, range {X.index.min().date()} .. {X.index.max().date()}")
        ep = _load_expected_predictors(target)
        print(f"  expected_predictors for {target}: {ep}")
        results = run_walk_forward_multihorizon(
            X, y, horizons=args.horizons,
            bsts_niter=args.niter, bsts_burn=args.burn,
            train_window=args.train_window, refit_every=args.refit_every,
            expected_predictors=ep,
        )
        for model_name, df in results.items():
            for h in args.horizons:
                df_h = df.xs(h, level="horizon") if isinstance(df.index, pd.MultiIndex) else df
                row = score_horizon(model_name, target, h, df_h, y)
                rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out)} rows = "
          f"{len(args.targets)} targets × 4 models × {len(args.horizons)} horizons)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
