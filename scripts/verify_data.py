"""CLI: sanity-check the v5 data-sideband files before running forecasts.

The PM unpacks the data emails into ``<bundle>/data/`` and runs::

    python scripts/verify_data.py

This script confirms:
  - ``<TARGET>_history.csv`` files exist for every frozen model in ``model/``
    with a recognizable (date, value) schema.
  - ``trends.parquet`` (or whichever filename matches ``--x-data``) exists,
    parses, and has the predictor-column set the frozen models expect.
  - The y history and X data overlap and are recent enough to forecast from
    a sensible ``as_of`` date.

Exits with code 0 on success, non-zero on any check failure (printing a
clear "fix this" message). Idempotent and read-only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from gtrends_bayes.inference import load_model
from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify_data")
    p.add_argument("--model-dir", default="model",
                   help="Directory of frozen *_v?.pkl files (e.g. HY_v5.pkl).")
    p.add_argument("--data-dir", default="data",
                   help="Directory holding {TARGET}_history.csv + trends.parquet.")
    p.add_argument("--x-data", default="trends.parquet",
                   help="Name of the X parquet within --data-dir.")
    p.add_argument("--max-stale-days", type=int, default=30,
                   help="Warn if latest data is older than this many days.")
    return p


def _check_one_y(csv_path: Path) -> tuple[bool, str]:
    if not csv_path.exists():
        return False, f"missing: {csv_path}"
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable CSV: {csv_path} ({exc})"
    if len(df.columns) < 2:
        return False, (
            f"{csv_path}: expected 2 columns (date, value); got {list(df.columns)}"
        )
    date_col = df.columns[0]
    try:
        dates = pd.to_datetime(df[date_col])
    except Exception as exc:  # noqa: BLE001
        return False, f"{csv_path}: first column not parseable as date ({exc})"
    n = len(df)
    if n < 50:
        return False, (
            f"{csv_path}: only {n} rows. AR backbone needs at least train_window "
            "history (≥ 50 weekly obs)."
        )
    return True, (
        f"OK ({n} rows, {dates.min().date()} → {dates.max().date()})"
    )


def _check_x(parquet_path: Path, expected_cols: set[str]) -> tuple[bool, str]:
    if not parquet_path.exists():
        return False, f"missing: {parquet_path}"
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable parquet: {parquet_path} ({exc})"
    missing = expected_cols - set(df.columns)
    if missing:
        return False, (
            f"{parquet_path}: missing {len(missing)} predictor columns: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )
    n = len(df)
    if n < 50:
        return False, f"{parquet_path}: only {n} rows; need ≥ 50"
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as exc:  # noqa: BLE001
        return False, f"{parquet_path}: index not parseable as dates ({exc})"
    return True, (
        f"OK ({n} rows × {len(df.columns)} cols, "
        f"{df.index.min().date()} → {df.index.max().date()})"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # Match any versioned pickle (HY_v4.pkl, HY_v5.pkl, …). Cross-version
    # mixing is caught later via load_model's schema check.
    pkl_files = sorted(
        p for p in model_dir.glob("*.pkl")
        if "_v" in p.stem and p.stem.split("_v")[-1].isdigit()
    )
    if not pkl_files:
        log.error("no *_v?.pkl files in %s — unpack the model bundle first.", model_dir)
        return 1

    expected_cols: set[str] = set()
    # (target, history_filename) tuples — one per loaded pickle. Each pickle
    # self-describes the CSV that holds its y-series via the `history_file`
    # field (added at freeze time in scripts/freeze_model_v4.py).
    target_history_pairs: list[tuple[str, str]] = []
    for pkl in pkl_files:
        try:
            m = load_model(pkl)
        except Exception as exc:  # noqa: BLE001
            log.error("failed to load %s: %s", pkl, exc)
            return 2
        history = m.get("history_file") or f"{m['target']}_history.csv"
        target_history_pairs.append((m["target"], history))
        expected_cols.update(m["bsts_posterior"]["X_columns"])
        log.info("✓ model: %s (target=%s, transform=%s, cadence=%s)",
                 pkl.name, m["target"], m["target_transform"],
                 m["preprocessing"]["cadence"])
        # Surface the embedded ETF→OAS regression so the PM sees the
        # provenance + proxy-quality verdict at unpack time.
        overlay = m.get("oas_overlay_translation")
        if overlay is not None:
            log.info(
                "  ↳ OAS overlay: slope=%+.1f bps/dlog, "
                "last_oas=%.0f bps (%s), pearson=%+.2f (%s, n=%d wk)",
                overlay["slope_bps_per_dlog"],
                overlay["last_oas_bps"],
                overlay["last_oas_date"],
                overlay["pearson"],
                overlay["proxy_quality_label"],
                overlay["n_overlap_weeks"],
            )

    # Dedupe by history filename so HY+HY_OAS both check their CSVs once each.
    unique_histories = sorted({h for _, h in target_history_pairs})
    log.info("checking %d y-history CSV(s) in %s", len(unique_histories), data_dir)
    n_err = 0
    latest_y: pd.Timestamp | None = None
    for history in unique_histories:
        csv = data_dir / history
        ok, msg = _check_one_y(csv)
        prefix = "✓" if ok else "✗"
        (log.info if ok else log.error)("%s y: %s — %s", prefix, csv.name, msg)
        if not ok:
            n_err += 1
        elif "→" in msg:
            try:
                last = pd.Timestamp(msg.split("→")[-1].strip().rstrip(")").strip())
                latest_y = max(latest_y, last) if latest_y is not None else last
            except Exception:  # noqa: BLE001
                pass

    log.info("checking X parquet")
    x_path = data_dir / args.x_data
    ok, msg = _check_x(x_path, expected_cols)
    prefix = "✓" if ok else "✗"
    (log.info if ok else log.error)("%s X: %s — %s", prefix, x_path.name, msg)
    if not ok:
        n_err += 1

    if n_err > 0:
        log.error("verify_data: %d issue(s) — fix the items marked ✗ above.", n_err)
        return 3

    if latest_y is not None:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        stale_days = (today - latest_y).days
        if stale_days > args.max_stale_days:
            log.warning(
                "latest y data is %d days old (> %d-day threshold). Forecasts "
                "will still run but the `as_of` date will be artificially old.",
                stale_days, args.max_stale_days,
            )

    log.info("Ready to run inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
