"""Workhorse for v3 Phases D.1, D.3, and E.

Three modes:
- ``--mode refit_sweep``  : vary ``--refit-every`` ∈ {1, 4, 13}; one ar_p;
                            horizon=1 (week). Drives D.1.
- ``--mode ar_bakeoff``   : StackedResidual only, vary ``--ar-p`` ∈ {1, 4};
                            fixed refit_every (D.1 winner). Drives D.3.
- ``--mode horizon_sweep``: 4 models × 6 weekly horizons (1d row appended
                            as caveat placeholder). Drives Phase E.

Reuses ``build_features``, ``_load_expected_predictors``, ``score_horizon``
from ``scripts/horizon_sweep.py``. Caches raw walk-forward parquets at
``data/processed/backtest/raw_v3/{target}_{slug}_re{re}_ar{p}.parquet``.

At end of ``--mode refit_sweep`` with ``--save-final-posterior``, also fits
one BSTS on the full window (no walk-forward) and pickles to
``data/processed/posterior/{target}_bsts_v3.pkl`` for v4 freeze + Phase F.

Usage
-----
    # D.1 — refit-cadence sweep
    PYTHONPATH=src python3 scripts/horizon_sweep_v3.py \
        --mode refit_sweep --refit-every 1 4 13 --ar-p 4 \
        --save-final-posterior \
        --out data/processed/backtest/refit_cadence.csv

    # D.3 — AR(1) vs AR(4) on StackedResidual
    PYTHONPATH=src python3 scripts/horizon_sweep_v3.py \
        --mode ar_bakeoff --ar-p 1 4 --refit-every 4 \
        --reuse-raw \
        --out data/processed/backtest/ar_bakeoff.csv

    # Phase E — 7-horizon sweep (with weekly→labels mapping)
    PYTHONPATH=src python3 scripts/horizon_sweep_v3.py \
        --mode horizon_sweep --refit-every 4 --ar-p 4 \
        --reuse-raw \
        --out data/processed/backtest/horizon_sweep_v3.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
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
    sys.path.insert(0, str(p / "scripts"))
    return p


MODELS_DEFAULT = ["BSTS (Trends)", "StackedResidual", "AR(p)", "Naive RW"]
"""v3 model set for Phase E. v2 used BSTS, AR(4), NaiveRW, AR(4)+VIX;
v3 swaps AR(p)+VIX for StackedResidual (the new headline)."""


# Horizon ladder: weekly units (what WalkForward expects) ↔ human labels.
# Trimmed for v3 final scope: 1d dropped (undefined for weekly target),
# 1y dropped (~18 non-overlap obs is noise-dominated). 5 horizons remain.
HORIZON_LADDER = [
    # (label, wf_weeks, business_days, caveat_or_None)
    ("1w",  1,    5,   None),
    ("2w",  2,    10,  None),
    ("1m",  4,    21,  None),
    ("1q",  13,   63,  None),
    ("6m",  26,   126, "small-N (~36 non-overlap)"),
]


def _slug(model_name: str) -> str:
    return (model_name.replace(" ", "_")
                      .replace("(", "")
                      .replace(")", "")
                      .replace("+", "plus"))


def _ar_aware(model_name: str) -> bool:
    """True if model's behavior changes with ar_p."""
    return model_name in {"StackedResidual", "AR(p)", "AR(p) + VIX"}


def _raw_path(raw_dir: Path, target: str, model: str,
              refit_every: int, ar_p: int) -> Path:
    """Cache key. Models that don't depend on ar_p use ar0 to avoid duplication."""
    p = ar_p if _ar_aware(model) else 0
    return raw_dir / f"{target}_{_slug(model)}_re{refit_every}_ar{p}.parquet"


def make_model_factory(model_name: str, ar_p: int, niter: int, burn: int,
                       expected_predictors: int):
    """Return a zero-arg factory that constructs the model fresh per refit."""
    from gtrends_bayes.models.baseline import AR_p, AR_VIX, NaiveRW
    from gtrends_bayes.models.bsts import BSTS
    from gtrends_bayes.models.stacked_residual import StackedResidualModel

    if model_name == "BSTS (Trends)":
        return lambda: BSTS(n_seasons=52,
                            expected_predictors=expected_predictors,
                            niter=niter, burn=burn, seed=42)
    if model_name == "StackedResidual":
        return lambda: StackedResidualModel(
            ar_p=ar_p,
            bsts_kwargs={"n_seasons": 52,
                         "expected_predictors": expected_predictors,
                         "niter": niter, "burn": burn, "seed": 42},
        )
    if model_name == "AR(p)":
        return lambda: AR_p(p=ar_p, seed=42)
    if model_name == "AR(p) + VIX":
        return lambda: AR_VIX(p=ar_p)
    if model_name == "Naive RW":
        return NaiveRW
    raise ValueError(f"unknown model: {model_name}")


def _x_input_for(model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """The X matrix WalkForward should pass to the factory for this model."""
    if model_name in {"BSTS (Trends)", "StackedResidual"}:
        return X
    if model_name == "AR(p) + VIX":
        return X[["vix"]] if "vix" in X.columns else X.iloc[:, :0]
    return X.iloc[:, :0]  # AR(p), Naive RW: no regressors


def _n_draws_for(model_name: str, base_n_draws: int) -> int:
    if model_name in {"BSTS (Trends)", "StackedResidual"}:
        return base_n_draws
    return 200


def run_one_cell(target: str, model_name: str, refit_every: int, ar_p: int,
                 horizons_weeks: list[int], niter: int, burn: int,
                 train_window: int, base_n_draws: int,
                 raw_dir: Path, reuse_raw: bool,
                 X: pd.DataFrame, y: pd.Series,
                 expected_predictors: int) -> tuple[pd.DataFrame, float]:
    """Run one (target, model, refit_every, ar_p) cell with given horizons.

    Returns
    -------
    (results_df, mean_fit_time_min)
        ``results_df`` is what WalkForward.run() returns (MultiIndex if
        multi-horizon, single Index otherwise). ``mean_fit_time_min`` is the
        per-refit average across the test window.
    """
    from gtrends_bayes.backtest.walk_forward import WalkForward
    from gtrends_bayes.models.bsts import reset_r_models

    path = _raw_path(raw_dir, target, model_name, refit_every, ar_p)
    timing_path = path.with_suffix(".timing.txt")
    if reuse_raw and path.exists():
        df = pd.read_parquet(path)
        # Recover timing if present.
        t = float("nan")
        if timing_path.exists():
            try:
                t = float(timing_path.read_text().strip())
            except ValueError:
                pass
        return df, t

    factory = make_model_factory(model_name, ar_p, niter, burn,
                                 expected_predictors)
    X_input = _x_input_for(model_name, X)
    n_draws = _n_draws_for(model_name, base_n_draws)

    wf = WalkForward(train_window=train_window, step=1,
                     horizons=horizons_weeks,
                     refit_every=refit_every, mode="backtest")
    t0 = time.perf_counter()
    df = wf.run(factory, X_input, y, n_draws=n_draws)
    elapsed_s = time.perf_counter() - t0

    n_steps = max(1, len(y) - train_window)
    n_refits = max(1, (n_steps + refit_every - 1) // refit_every)
    mean_fit_s = elapsed_s / n_refits
    mean_fit_min = mean_fit_s / 60.0

    if model_name in {"BSTS (Trends)", "StackedResidual"}:
        reset_r_models()

    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    timing_path.write_text(f"{mean_fit_min:.4f}\n")
    return df, mean_fit_min


def score_cell(model_name: str, target: str, horizons_weeks: list[int],
               result_df: pd.DataFrame, y: pd.Series,
               mean_fit_time_min: float, refit_every: int, ar_p: int,
               target_transform: str = "log_diff") -> list[dict]:
    """Score one cell's result_df across its horizons. Returns one row per h."""
    from horizon_sweep import score_horizon  # type: ignore

    rows: list[dict] = []
    single_horizon = len(horizons_weeks) == 1 or \
        not isinstance(result_df.index, pd.MultiIndex)
    for h in horizons_weeks:
        if single_horizon:
            df_h = result_df
        else:
            df_h = result_df.xs(h, level="horizon")
        row = score_horizon(model_name, target, h, df_h, y)
        row.update({
            "refit_every": refit_every,
            "ar_p": ar_p,
            "mean_fit_time_min": round(mean_fit_time_min, 3),
            "target_transform": target_transform,
        })
        rows.append(row)
    return rows


def _compute_rmse_level(target: str, h_weeks: int, df_h: pd.DataFrame,
                        y_levels: pd.Series) -> float:
    """RMSE re-aggregated to level space via log_diff inverse_transform.

    For ETF targets in v2.1, ``y`` is already the level-space price; the model
    forecasts level prices (BSTS) or level prices (AR_p on levels). To match
    the plan's ``RMSE_level`` semantics, we compute RMSE in price space —
    which is exactly the per-row RMSE already produced by ``score_horizon``.
    For OAS targets in future v3.x where y is transformed, this would call
    ``TargetTransform("log_diff").inverse_transform``. For now: same number
    as RMSE_transform.
    """
    common = df_h.index.intersection(y_levels.index)
    y_true = y_levels.loc[common]
    y_pred = df_h.loc[common, "y_pred_mean"]
    return float(np.sqrt(((y_true - y_pred) ** 2).mean())) if len(common) else float("nan")


def fit_final_posterior(target: str, niter: int, burn: int,
                        expected_predictors: int,
                        posterior_dir: Path) -> Path:
    """Fit one BSTS on the full window (no walk-forward); pickle for v4 + F.

    Pickle schema (compatible with scripts/freeze_model_v4.py):
        {
          "target": str,
          "y": pd.Series,
          "X_columns": list[str],
          "inclusion_probs": pd.Series,
          "coefficient_summary": pd.DataFrame,
          "component_bands": dict[str, pd.DataFrame],
          "in_sample_fit_median": pd.Series,
          "niter": int, "burn": int, "expected_predictors": int,
        }
    """
    from gtrends_bayes.models.bsts import BSTS, reset_r_models
    from horizon_sweep import build_features

    print(f"\n--- fitting final BSTS posterior for {target} ---")
    X, y = build_features(target)
    print(f"  full X={X.shape}, y={y.shape}")

    bsts = BSTS(n_seasons=52, expected_predictors=expected_predictors,
                niter=niter, burn=burn, seed=42)
    bsts.fit(y, X)

    bands = bsts.component_bands()
    in_sample_median = sum(b["q_med"] for b in bands.values())

    payload = {
        "target": target,
        "y": y,
        "X_columns": list(X.columns),
        "inclusion_probs": bsts.inclusion_probabilities(),
        "coefficient_summary": bsts.coefficient_summary(),
        "component_bands": bands,
        "in_sample_fit_median": in_sample_median,
        "niter": niter,
        "burn": burn,
        "expected_predictors": expected_predictors,
    }
    posterior_dir.mkdir(parents=True, exist_ok=True)
    out_path = posterior_dir / f"{target}_bsts_v3.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    reset_r_models()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(prog="horizon_sweep_v3")
    parser.add_argument("--mode", required=True,
                        choices=["refit_sweep", "ar_bakeoff", "horizon_sweep"])
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--refit-every", type=int, nargs="+", default=[4])
    parser.add_argument("--ar-p", type=int, nargs="+", default=[4])
    parser.add_argument("--niter", type=int, default=900)
    parser.add_argument("--burn", type=int, default=150)
    parser.add_argument("--train-window", type=int, default=260)
    parser.add_argument("--n-draws", type=int, default=400)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override the model set (default 4 v3 models).")
    parser.add_argument("--raw-out", type=str,
                        default="data/processed/backtest/raw_v3")
    parser.add_argument("--out", type=str)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--save-final-posterior", action="store_true",
                        help="(refit_sweep only) fit + pickle full-window BSTS.")
    parser.add_argument("--posterior-dir", type=str,
                        default="data/processed/posterior")
    args = parser.parse_args()

    project_root()
    from horizon_sweep import _load_expected_predictors, build_features

    raw_dir = Path(args.raw_out)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mode-specific defaults --------------------------------------
    if args.mode == "refit_sweep":
        models = args.models or MODELS_DEFAULT
        horizons_weeks = [1]
        ar_p_values = args.ar_p  # typically [4]
        refit_values = args.refit_every  # typically [1, 4, 13]
        default_out = "data/processed/backtest/refit_cadence.csv"
    elif args.mode == "ar_bakeoff":
        models = args.models or ["StackedResidual"]
        horizons_weeks = [1]
        ar_p_values = args.ar_p  # typically [1, 4]
        refit_values = args.refit_every[:1]  # only the first (D.1 winner)
        default_out = "data/processed/backtest/ar_bakeoff.csv"
    elif args.mode == "horizon_sweep":
        models = args.models or MODELS_DEFAULT
        horizons_weeks = [hw for (_, hw, _, _) in HORIZON_LADDER if hw is not None]
        ar_p_values = args.ar_p[:1]  # D.3 winner
        refit_values = args.refit_every[:1]  # D.1 winner
        default_out = "data/processed/backtest/horizon_sweep_v3.csv"

    out_path = Path(args.out) if args.out else Path(default_out)
    print(f"mode={args.mode}, targets={args.targets}, models={models}, "
          f"refit_every={refit_values}, ar_p={ar_p_values}, "
          f"horizons_weeks={horizons_weeks}")

    rows: list[dict] = []
    for target in args.targets:
        print(f"\n=== {target} ===")
        X, y = build_features(target)
        ep = _load_expected_predictors(target)
        print(f"  X={X.shape}, y={y.shape}, "
              f"range {X.index.min().date()}..{X.index.max().date()}, "
              f"expected_predictors={ep}")

        for refit_every in refit_values:
            for ar_p in ar_p_values:
                for model_name in models:
                    # Skip cells where ar_p doesn't apply but we're iterating ar_p
                    if not _ar_aware(model_name) and ar_p != ar_p_values[0]:
                        continue
                    print(f"  >> {model_name} | refit_every={refit_every} "
                          f"| ar_p={ar_p}")
                    df, mean_min = run_one_cell(
                        target, model_name, refit_every, ar_p,
                        horizons_weeks, args.niter, args.burn,
                        args.train_window, args.n_draws,
                        raw_dir, args.reuse_raw,
                        X, y, ep,
                    )
                    print(f"     mean_fit_time = {mean_min:.3f} min")
                    cell_rows = score_cell(
                        model_name, target, horizons_weeks, df, y,
                        mean_min, refit_every, ar_p,
                    )
                    # Phase E: add caveat + RMSE_level + 1d placeholder
                    if args.mode == "horizon_sweep":
                        for r in cell_rows:
                            h = r["horizon"]
                            label = next(lbl for (lbl, hw, _, _)
                                         in HORIZON_LADDER if hw == h)
                            bd = next(bd for (_, hw, bd, _)
                                      in HORIZON_LADDER if hw == h)
                            cav = next(cv for (_, hw, _, cv)
                                       in HORIZON_LADDER if hw == h)
                            r["horizon_label"] = label
                            r["horizon_bd"] = bd
                            r["caveat"] = cav or ""
                            if not isinstance(df.index, pd.MultiIndex):
                                df_h = df
                            else:
                                df_h = df.xs(h, level="horizon")
                            r["RMSE_level"] = round(
                                _compute_rmse_level(target, h, df_h, y), 4)
                    rows.extend(cell_rows)

        # End of refit_sweep mode: optionally pickle final-window BSTS
        if args.mode == "refit_sweep" and args.save_final_posterior:
            fit_final_posterior(target, args.niter, args.burn, ep,
                                Path(args.posterior_dir))

    df_out = pd.DataFrame(rows)
    # Phase E sorting: by target, model, horizon_bd
    if args.mode == "horizon_sweep" and "horizon_bd" in df_out.columns:
        df_out = df_out.sort_values(["target", "model", "horizon_bd"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df_out)} rows)")
    print(df_out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
