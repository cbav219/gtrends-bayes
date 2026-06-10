"""CLI entry point for one-off forecasts.

Run from the unpacked bundle root::

    python -m gtrends_bayes.inference \\
      --model-path model/HY_v5.pkl \\
      --horizon 1m \\
      --as-of 2026-05-15 \\
      --y-data data/HY_history.csv \\
      --x-data data/trends.parquet

Output is a JSON dump of the forecast dict (see
:func:`gtrends_bayes.inference.forecast` for the schema). For most PM
workflows, ``scripts/example_forecast.py`` is the easier entry point — it
iterates over both targets and the 4-horizon ladder with one command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .forecast import HORIZON_BD, forecast
from .load import load_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gtrends_bayes.inference",
        description=(
            "Run a single horizon-step-ahead forecast from a frozen v5 "
            "gtrends-bayes model. Forecasts HY/IG corporate-bond ETF "
            "prices (HYG / LQD) as proxies for HY/IG OAS spreads. See "
            "USAGE.md inside the bundle for full guidance."
        ),
    )
    p.add_argument(
        "--model-path", required=True,
        help="Path to the frozen model pickle (e.g. model/HY_v5.pkl).",
    )
    p.add_argument(
        "--horizon", required=True,
        help=(
            "Forecast horizon. Use one of "
            f"{sorted(HORIZON_BD)} (recommended PM ladder: 1w / 1m / 1q) "
            "or an explicit integer business-day count."
        ),
    )
    p.add_argument(
        "--as-of", required=True,
        help=(
            "Decision date (YYYY-MM-DD). Forecast is made *from* this "
            "date; y/x history must cover everything up to and including "
            "this date."
        ),
    )
    p.add_argument(
        "--y-data", required=True,
        help=(
            "Path to a 2-column CSV (date, value) with the target "
            "history. Inside the v5 data sideband: data/HY_history.csv "
            "or data/IG_history.csv."
        ),
    )
    p.add_argument(
        "--x-data", required=True,
        help=(
            "Path to the 41-column preprocessed Trends parquet. Inside "
            "the v5 data sideband: data/trends.parquet."
        ),
    )
    p.add_argument(
        "--n-draws", type=int, default=1000,
        help="Posterior draws for the credible band (default 1000).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default 42).",
    )
    p.add_argument(
        "--output", default="-",
        help=(
            "Where to write the JSON output. Default '-' = stdout. Pass "
            "a path like forecast.json to capture to file."
        ),
    )
    p.add_argument(
        "--target", default=None,
        help=(
            "Optional safety check: error out if the loaded model's "
            "'target' field doesn't match this string (e.g. 'HY'). "
            "Catches the case of loading the wrong pickle by accident."
        ),
    )
    return p


def _load_y(path: str) -> pd.Series:
    """Read a 2-column CSV (date, value) → pandas.Series indexed by date."""
    df = pd.read_csv(path)
    if len(df.columns) < 2:
        raise ValueError(
            f"y-data {path} must have at least 2 columns (date, value); "
            f"got {list(df.columns)}"
        )
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)[val_col].dropna()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = load_model(args.model_path)

    # Sanity check: if the PM passed --target, confirm the loaded pickle
    # really is for that target. Cheap insurance against bundle mix-ups.
    if args.target and model["target"] != args.target:
        raise SystemExit(
            f"--target={args.target} but model['target']={model['target']!r} "
            "(double-check you loaded the right pickle)"
        )

    y_series = _load_y(args.y_data)
    x = pd.read_parquet(args.x_data)
    x.index = pd.to_datetime(x.index)

    # Allow horizon as either a label ('1m') or a bare int ('21').
    horizon: str | int = args.horizon
    if horizon not in HORIZON_BD:
        try:
            horizon = int(horizon)
        except ValueError as exc:
            raise SystemExit(
                f"--horizon must be one of {sorted(HORIZON_BD)} "
                f"or an int business-day count; got {args.horizon!r}"
            ) from exc

    result = forecast(
        model, horizon, pd.Timestamp(args.as_of),
        y_series, x,
        n_draws=args.n_draws, seed=args.seed,
    )

    payload = json.dumps(result, indent=2, default=str)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
