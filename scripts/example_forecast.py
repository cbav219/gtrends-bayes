"""One-pager runnable example: load HY_v5 + synthetic data → forecast.

Demonstrates the v5 forecast API end-to-end. When data sideband files
are present, swap the synthetic ``y_history`` / ``x_latest`` for the real
ones. Prints a readable report for the 4 PM-facing horizons.

    cd <unpacked bundle>
    python scripts/example_forecast.py            # synthetic
    python scripts/example_forecast.py --real     # uses data/*_history.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gtrends_bayes.inference import forecast, load_model

PM_HORIZONS = ["1w", "2w", "1m", "1q"]


def _synthetic_inputs(model: dict, n_obs: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """Build synthetic y_history + x_latest aligned with the frozen model.

    y_history follows a slow random walk near the AR backbone's intercept;
    x_latest is N(0, 0.5) on each predictor — enough for the inference
    module to run end-to-end without real data.
    """
    cadence = model["preprocessing"]["cadence"]
    freq = "B" if cadence == "daily" else "W-SUN"
    idx = pd.date_range("2023-01-01", periods=n_obs, freq=freq)
    rng = np.random.default_rng(0)
    # Use a midpoint level so HY hovers near $80, IG near $110.
    base = 80.0 if model["target"] == "HY" else 110.0
    y = pd.Series(
        base * np.cumprod(1 + rng.normal(0, 0.005, size=n_obs)),
        index=idx, name=model["target"],
    )
    cols = model["bsts_posterior"]["X_columns"]
    x = pd.DataFrame(
        rng.normal(0, 0.3, size=(n_obs, len(cols))),
        index=idx, columns=cols,
    )
    return y, x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="example_forecast")
    p.add_argument("--model-dir", default="model",
                   help="Directory of frozen *_v?.pkl files (e.g. HY_v5.pkl).")
    p.add_argument("--real", action="store_true",
                   help="Use data/*_history.csv + data/trends.parquet instead of synthetic.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-draws", type=int, default=500)
    return p


def _load_real_y(data_dir: Path, history_file: str, target: str) -> pd.Series:
    """Load a target history CSV. The bundle ships
    ``HY_history.csv`` / ``IG_history.csv`` (ETF prices, USD) and the OAS
    variants ``HY_OAS_history.csv`` / ``IG_OAS_history.csv`` (bps).
    """
    csv = data_dir / history_file
    df = pd.read_csv(csv)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    return df.set_index(df.columns[0])[df.columns[1]].dropna().rename(target)


def _target_header(target: str) -> str:
    """One-line description for the column header above each block."""
    return {
        "HY":     "HYG ETF (USD) — proxy for HY OAS",
        "IG":     "LQD ETF (USD) — proxy for IG OAS",
        "HY_OAS": "HY OAS-direct (bps, ICE BAML BAMLH0A0HYM2)",
        "IG_OAS": "IG OAS-direct (bps, ICE BAML BAMLC0A0CM)",
    }.get(target, target)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    # Match any versioned pickle (HY_v4.pkl, HY_v5.pkl, HY_OAS_v5.pkl, …).
    pkl_files = sorted(
        p for p in model_dir.glob("*.pkl")
        if "_v" in p.stem and p.stem.rsplit("_v", 1)[-1].isdigit()
    )
    if not pkl_files:
        print(f"no *_v?.pkl in {model_dir}; have you unpacked the bundle?",
              file=sys.stderr)
        return 1

    if args.real:
        x_real = pd.read_parquet(Path(args.data_dir) / "trends.parquet")
        x_real.index = pd.to_datetime(x_real.index)

    print(f"v5 inference example ({'real data' if args.real else 'synthetic data'})")
    print("=" * 88)
    for pkl in pkl_files:
        model = load_model(pkl)
        target = model["target"]
        cadence = model["preprocessing"]["cadence"]
        transform = model["target_transform"]
        history_file = model.get("history_file", f"{target}_history.csv")

        print(f"\n{target}  [{_target_header(target)}]")
        print(f"        transform={transform}, cadence={cadence}, "
              f"α={model['conformal_alpha']:.3f}")
        print("-" * 88)

        if args.real:
            y = _load_real_y(Path(args.data_dir), history_file, target)
            # Restrict X to dates ≤ last y observation.
            x = x_real.loc[x_real.index <= y.index.max()]
        else:
            y, x = _synthetic_inputs(model)

        as_of = y.index.max()
        last_level = float(y.iloc[-1])
        # "Level" units differ by target: ETF = USD, OAS-direct = bps.
        level_units = "bps" if target.endswith("_OAS") else "USD"
        print(f"as_of: {as_of.date()}    last observed: {last_level:.2f} {level_units}")
        print(f"{'horizon':<8} {'Δ median':>10} {'Δ 5%':>9} {'Δ 95%':>9} "
              f"{'level median':>13} {'level 5%':>10} {'level 95%':>10}")

        results: list[tuple[str, dict]] = []
        for h in PM_HORIZONS:
            r = forecast(model, h, as_of, y, x, n_draws=args.n_draws, seed=42)
            results.append((h, r))
            # For levels-transform models, "Δ median" is just (level - last_level).
            if transform == "levels":
                delta_med = r["level_median"] - last_level
                delta_lo = r["level_band"][0] - last_level
                delta_hi = r["level_band"][1] - last_level
            else:
                delta_med, delta_lo, delta_hi = r["median"], r["q05"], r["q95"]
            print(f"{h:<8} {delta_med:>+10.4f} {delta_lo:>+9.4f} {delta_hi:>+9.4f} "
                  f"{r['level_median']:>13.2f} "
                  f"{r['level_band'][0]:>10.2f} {r['level_band'][1]:>10.2f}")

        # If the model carries an OAS-overlay translation, print a second
        # block showing the ETF forecast re-expressed in OAS bps.
        overlay = model.get("oas_overlay_translation")
        if overlay is not None and any("oas_implied_median" in r for _, r in results):
            print(f"\n        OAS-implied (via ETF↔OAS regression): "
                  f"slope={overlay['slope_bps_per_dlog']:+.1f} bps/dlog, "
                  f"pearson={overlay['pearson']:+.2f} ({overlay['proxy_quality_label']}), "
                  f"n={overlay['n_overlap_weeks']} wk overlap")
            print(f"        last OAS anchor: {overlay['last_oas_bps']:.0f} bps "
                  f"({overlay['last_oas_date']})")
            print(f"{'horizon':<8} {'Δ bps median':>13} {'Δ bps low':>11} "
                  f"{'Δ bps high':>11} {'OAS median':>11} {'OAS low':>10} "
                  f"{'OAS high':>10}")
            anchor = float(overlay["last_oas_bps"])
            for h, r in results:
                if "oas_implied_median" not in r:
                    continue
                med = r["oas_implied_median"]
                lo, hi = r["oas_implied_band"]
                print(f"{h:<8} {med - anchor:>+13.2f} {lo - anchor:>+11.2f} "
                      f"{hi - anchor:>+11.2f} {med:>11.1f} {lo:>10.1f} {hi:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
