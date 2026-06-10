"""Evaluate the published Trends Risk Index against the v1 plan's four tests:

1. **Granger causality** — does ``risk_index_{t-1}`` Granger-cause
   ``Δlog y_t`` incremental to ``Δlog VIX_{t-1}``?
2. **Quantile portfolios** — bucket weeks by index quintile; report mean
   forward Δlog y by bucket.
3. **Crisis recall** — was the index in the top decile in the 4 weeks
   preceding 2020-03 COVID, 2022-09 UK gilt, 2023-03 SVB?
4. **Lead/lag vs VIX** — cross-correlation of ``zscore_t`` vs
   ``Δlog VIX_{t+k}`` for ``k ∈ [-4, +4]``. Index that *leads* VIX is
   uniquely valuable.

Output: ``data/processed/risk_index/_evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
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


def granger_test(
    y_change: pd.Series,
    risk_index_z: pd.Series,
    vix_change: pd.Series,
    max_lag: int = 4,
) -> dict:
    """OLS-based Granger test: does risk_index add explanatory power for y over VIX?

    Fits two regressions on the common date range:
        restricted: Δy_t ~ const + Σ Δy_{t-k} + Σ Δvix_{t-k}      for k=1..max_lag
        full:       restricted + Σ risk_index_{t-k}              for k=1..max_lag
    Returns the F-statistic, p-value, and ΔR² of the additional regressors.
    """
    from scipy.stats import f as f_dist

    df = pd.concat({"y": y_change, "z": risk_index_z, "vix": vix_change}, axis=1).dropna()
    rows = []
    for col in ("y", "z", "vix"):
        for k in range(1, max_lag + 1):
            rows.append((f"{col}_lag{k}", df[col].shift(k)))
    lagged = pd.concat({name: s for name, s in rows}, axis=1).dropna()
    target = df["y"].loc[lagged.index]

    own_vix_cols = [c for c in lagged.columns if c.startswith("y_lag") or c.startswith("vix_lag")]
    z_cols = [c for c in lagged.columns if c.startswith("z_lag")]

    def _ols_rss(X: pd.DataFrame, y: pd.Series) -> tuple[float, int, int]:
        X = np.column_stack([np.ones(len(X)), X.values])
        coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        resid = y.values - X @ coef
        return float((resid ** 2).sum()), X.shape[1], len(y)

    rss_r, k_r, n = _ols_rss(lagged[own_vix_cols], target)
    rss_f, k_f, _ = _ols_rss(lagged[own_vix_cols + z_cols], target)
    df_num = k_f - k_r
    df_den = n - k_f
    if rss_f <= 0 or df_den <= 0 or df_num <= 0:
        return {"f_stat": float("nan"), "p_value": float("nan"),
                "delta_r2": float("nan"), "n_obs": int(n)}
    f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    p = 1.0 - f_dist.cdf(f_stat, df_num, df_den)
    tss = float(((target - target.mean()) ** 2).sum())
    delta_r2 = (rss_r - rss_f) / tss if tss > 0 else float("nan")
    return {
        "f_stat": float(f_stat),
        "p_value": float(p),
        "delta_r2": float(delta_r2),
        "n_obs": int(n),
        "max_lag": int(max_lag),
    }


def quantile_portfolios(risk_index_z: pd.Series, y_change_forward: pd.Series,
                        n_quantiles: int = 5) -> dict:
    """Bucket weeks by risk-index quintile; return mean forward Δlog y per bucket."""
    df = pd.concat({"z": risk_index_z, "fwd": y_change_forward}, axis=1).dropna()
    if len(df) < n_quantiles * 5:
        return {"by_quantile": [], "monotone": False}
    df["q"] = pd.qcut(df["z"], q=n_quantiles, labels=False, duplicates="drop")
    grp = df.groupby("q")["fwd"].agg(["mean", "std", "count"])
    grp = grp.reset_index().rename(columns={"q": "quantile"})
    means = grp["mean"].values
    monotone = bool(np.all(np.diff(means) >= 0)) or bool(np.all(np.diff(means) <= 0))
    return {
        "by_quantile": grp.to_dict(orient="records"),
        "monotone": monotone,
        "spread_top_minus_bottom": float(means[-1] - means[0]),
    }


def crisis_recall(risk_index_z: pd.Series, crises: dict[str, pd.Timestamp],
                  preceding_weeks: int = 4, top_decile_threshold: float = None) -> dict:
    """Check whether the index was in the top decile in the 4 weeks before each crisis."""
    valid = risk_index_z.dropna()
    threshold = (
        top_decile_threshold if top_decile_threshold is not None
        else float(valid.quantile(0.90))
    )
    out = {"top_decile_threshold": float(threshold), "by_crisis": {}}
    for label, anchor in crises.items():
        window_start = anchor - pd.Timedelta(weeks=preceding_weeks)
        window = valid.loc[window_start:anchor]
        max_in_window = float(window.max()) if len(window) else float("nan")
        any_top_decile = bool((window >= threshold).any())
        out["by_crisis"][label] = {
            "anchor": anchor.date().isoformat(),
            "window_start": window_start.date().isoformat(),
            "n_weeks_in_window": int(len(window)),
            "max_zscore_in_window": (max_in_window if not np.isnan(max_in_window) else None),
            "in_top_decile": any_top_decile,
        }
    out["recall"] = (
        sum(1 for d in out["by_crisis"].values() if d["in_top_decile"])
        / max(1, len(out["by_crisis"]))
    )
    return out


def cross_correlation(risk_index_z: pd.Series, vix_change: pd.Series,
                      max_k: int = 4) -> dict:
    """corr(risk_index_t, Δlog VIX_{t+k}) for k ∈ [-max_k, +max_k]. Positive k means VIX
    moves AFTER the index — i.e. the index leads."""
    df = pd.concat({"z": risk_index_z, "v": vix_change}, axis=1).dropna()
    out = {}
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            corr = df["z"].corr(df["v"].shift(-k))
        else:
            corr = df["z"].corr(df["v"].shift(-k))
        out[k] = float(corr) if pd.notna(corr) else None
    # Find the k with max absolute correlation.
    valid = {k: v for k, v in out.items() if v is not None}
    best_k = max(valid, key=lambda k: abs(valid[k])) if valid else None
    return {
        "by_lag": out,
        "best_lag": int(best_k) if best_k is not None else None,
        "best_corr": float(valid[best_k]) if best_k is not None else None,
        "interpretation": (
            "index LEADS VIX" if best_k is not None and best_k > 0
            else "VIX LEADS index" if best_k is not None and best_k < 0
            else "contemporaneous"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluate_risk_index")
    parser.add_argument("--targets", nargs="+", default=["HY", "IG"])
    parser.add_argument("--risk-index-dir", default="data/processed/risk_index")
    parser.add_argument("--targets-dir", default="data/raw/targets")
    parser.add_argument("--out", default="data/processed/risk_index/_evaluation.json")
    parser.add_argument("--max-granger-lag", type=int, default=4)
    args = parser.parse_args()

    project_root()
    from gtrends_bayes.features.trends_risk_index import crisis_windows

    crises = crisis_windows()

    # Load shared series.
    targets_dir = Path(args.targets_dir)
    vix_path = targets_dir / "vix.parquet"
    if not vix_path.exists():
        print(f"VIX cache not found at {vix_path}", file=sys.stderr)
        return 1
    vix_levels = pd.read_parquet(vix_path).iloc[:, 0]
    vix_levels.index = pd.DatetimeIndex(vix_levels.index)
    vix_change = np.log(vix_levels).diff().rename("vix_dlog")

    by_target: dict[str, dict] = {}
    for target in args.targets:
        ri_path = Path(args.risk_index_dir) / f"{target}_trends_risk_index.parquet"
        y_path = targets_dir / f"{target}.parquet"
        if not ri_path.exists():
            print(f"skipping {target}: {ri_path} missing")
            continue
        if not y_path.exists():
            print(f"skipping {target}: {y_path} missing")
            continue
        ri = pd.read_parquet(ri_path)
        ri.index = pd.DatetimeIndex(ri.index)
        z = ri["zscore_5y"].rename("zscore_5y")

        y = pd.read_parquet(y_path).iloc[:, 0]
        y.index = pd.DatetimeIndex(y.index)
        y_change = np.log(y).diff().rename(f"{target}_dlog")

        # Forward 1-week change for the quantile-portfolio test.
        y_fwd = y_change.shift(-1)

        # 1. Granger.
        granger = granger_test(y_change, z, vix_change, max_lag=args.max_granger_lag)
        # 2. Quantile portfolios.
        quants = quantile_portfolios(z, y_fwd, n_quantiles=5)
        # 3. Crisis recall.
        crecall = crisis_recall(z, crises)
        # 4. Lead/lag vs VIX.
        cc = cross_correlation(z, vix_change, max_k=4)

        by_target[target] = {
            "n_obs": int(z.dropna().shape[0]),
            "zscore_min": float(z.dropna().min()),
            "zscore_max": float(z.dropna().max()),
            "granger": granger,
            "quantile_portfolios": quants,
            "crisis_recall": crecall,
            "cross_correlation_vs_vix": cc,
        }
        print(f"\n=== {target} ===")
        print(f"  Granger F={granger['f_stat']:.2f}, p={granger['p_value']:.4f}, "
              f"ΔR²={granger['delta_r2']:.4f}")
        print(f"  Quantile spread (top - bottom): {quants['spread_top_minus_bottom']:+.5f} "
              f"(monotone={quants['monotone']})")
        print(f"  Crisis recall: {crecall['recall']:.0%}  "
              f"({sum(d['in_top_decile'] for d in crecall['by_crisis'].values())}/{len(crecall['by_crisis'])})")
        print(f"  Lead/lag vs VIX: best k={cc['best_lag']}, corr={cc['best_corr']:.3f}  "
              f"({cc['interpretation']})")

    out_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "by_target": by_target,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
