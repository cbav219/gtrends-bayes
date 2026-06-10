"""Build the v4 data-sideband zip the PM needs to run real forecasts.

Produces ``dist/v4/gtrends-bayes-v4-data.zip`` containing:

    data/
      HY_history.csv      — date,price (weekly Sunday-aligned, HYG ETF)
      IG_history.csv      — date,price (weekly Sunday-aligned, LQD ETF)
      trends.parquet      — preprocessed X matrix (957 × 43 cols)
      HY_OAS_history.csv  — date,oas_bps (FRED ICE BAML, 2023-05+ reference)
      IG_OAS_history.csv  — same for IG
      README.md           — what's in the zip

The PM unpacks this into the unpacked v4 bundle's ``data/`` directory and
runs ``scripts/example_forecast.py --real`` to get real forecasts.

Usage
-----
    PYTHONPATH=src python3 scripts/build_data_sideband.py
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
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


def _etf_history_csv(target: str, src_dir: Path) -> bytes:
    """Weekly ETF price → CSV bytes (date,price)."""
    p = src_dir / f"{target}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    s = pd.read_parquet(p).iloc[:, 0]
    s.index = pd.DatetimeIndex(s.index)
    df = s.rename("price").to_frame()
    df.index.name = "date"
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


def _oas_history_csv(target: str) -> bytes:
    """FRED OAS → CSV bytes (date,oas_bps). Uses the cached parquet from
    scripts/oas_overlay_v3.py (which has already ×100 to bps).
    """
    p = Path(f"data/processed/oas_overlay/{target}_OAS_weekly.parquet")
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; run scripts/oas_overlay_v3.py first"
        )
    s = pd.read_parquet(p).iloc[:, 0]
    s.index = pd.DatetimeIndex(s.index)
    df = s.rename("oas_bps").to_frame()
    df.index.name = "date"
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


def _build_trends_parquet(x_columns: list[str]) -> bytes:
    """Re-build the processed weekly X matrix matching the v4 model schema."""
    from gtrends_bayes.config import PredictorsConfig, TargetsConfig
    from gtrends_bayes.data.loader import (
        load_predictor_samples,
        predictor_classes,
    )
    from gtrends_bayes.features.library import (
        add_market_controls,
        drop_low_quality_columns,
        load_market_controls,
    )
    from gtrends_bayes.preprocessing.pipeline import Pipeline

    pred_cfg = PredictorsConfig.from_yaml("config/predictors.yaml")
    tgt_cfg = TargetsConfig.from_yaml("config/targets.yaml")
    long_df = load_predictor_samples(pred_cfg, rename_to_human=True)
    classes = predictor_classes(pred_cfg, rename_to_human=True)
    pipe = Pipeline(classes=classes, hp_lambda=129_600,
                    weighted_neighbor=True)
    processed = pipe.fit_transform(long_df)
    processed = drop_low_quality_columns(processed, nan_threshold=0.5)

    # Match the v4 model's expected X_columns. Fill missing ones with NaN
    # (the inference module's preprocess.py handles missing columns).
    controls = load_market_controls(tgt_cfg)
    processed_with_ctrls, _ = add_market_controls(processed, controls)
    X = processed_with_ctrls.reindex(columns=x_columns)
    buf = io.BytesIO()
    X.to_parquet(buf, engine="pyarrow", compression="snappy")
    return buf.getvalue()


def _readme_text(bundle_version: str = "v4") -> str:
    return f"""# gtrends-bayes {bundle_version} — Data Sideband

Generated {datetime.now(timezone.utc).isoformat()}

## What's in this zip

| File | Content | Source |
|---|---|---|
| `HY_history.csv` | Weekly HYG ETF price (proxy for HY OAS) | yfinance, Sunday-anchored |
| `IG_history.csv` | Weekly LQD ETF price (proxy for IG OAS) | yfinance, Sunday-anchored |
| `trends.parquet` | Preprocessed 43-column Google Trends matrix | Pulled + OECD-pipeline'd locally |
| `HY_OAS_history.csv` | Weekly HY OAS (bps) — reference only | FRED `BAMLH0A0HYM2`, 2023-05+ |
| `IG_OAS_history.csv` | Weekly IG OAS (bps) — reference only | FRED `BAMLC0A0CM`, 2023-05+ |

## How to use

Unpack this zip into `<gtrends-bayes-{bundle_version}>/data/`:

```bash
cd gtrends-bayes-{bundle_version}
unzip ../gtrends-bayes-{bundle_version}-data.zip
python scripts/verify_data.py    # expect ✓ on all rows
python scripts/example_forecast.py --real    # real forecasts
```

## Notes

- The model forecasts the **ETF price** (HYG / LQD) — `transform=levels`.
  ETF prices serve as proxies for HY/IG OAS spreads. See the parent
  bundle's `USAGE.md` for the proxy-quality caveat (HYG↔HY_OAS Pearson
  ≈ −0.69, LQD↔IG_OAS ≈ −0.24).
- The OAS CSVs are 2023-05+ only — too short to retrain BSTS but useful
  for current-level reference. If you ingest longer-history OAS from
  Bloomberg, the project's `IMPLEMENTATION_PLAN_v3.md` documents the
  retrain branch.
"""


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_data_sideband")
    parser.add_argument(
        "--bundle-version", default="v4",
        help="Bundle version tag (e.g. 'v4', 'v5'). Drives the default "
             "model-dir, the model-pickle filename, and the output zip name.",
    )
    parser.add_argument(
        "--bundle-model-dir", default=None,
        help="Where the frozen *_{version}.pkl files live. "
             "Defaults to dist/{bundle-version}/model.",
    )
    parser.add_argument("--etf-cache-dir", default="data/raw/targets")
    parser.add_argument(
        "--out", default=None,
        help="Output zip path. Defaults to "
             "dist/{bundle-version}/gtrends-bayes-{bundle-version}-data.zip.",
    )
    args = parser.parse_args()

    project_root()
    import pickle

    bundle_version = args.bundle_version
    # Late-resolve defaults that depend on bundle_version.
    if args.bundle_model_dir is None:
        args.bundle_model_dir = f"dist/{bundle_version}/model"
    if args.out is None:
        args.out = f"dist/{bundle_version}/gtrends-bayes-{bundle_version}-data.zip"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull the canonical X_columns from the frozen HY model
    # (the IG model has the same X_columns by construction).
    hy_pkl = Path(args.bundle_model_dir) / f"HY_{bundle_version}.pkl"
    if not hy_pkl.exists():
        print(
            f"ERROR: {hy_pkl} not found. Run "
            f"scripts/freeze_model_v4.py --bundle-version {bundle_version} first.",
            file=sys.stderr,
        )
        return 1
    hy_model = pickle.load(open(hy_pkl, "rb"))
    x_columns = list(hy_model["bsts_posterior"]["X_columns"])
    print(f"X_columns from {hy_pkl.name}: {len(x_columns)} predictors")

    print("\nbuilding HY/IG ETF history CSVs...")
    etf_dir = Path(args.etf_cache_dir)
    hy_csv = _etf_history_csv("HY", etf_dir)
    ig_csv = _etf_history_csv("IG", etf_dir)
    print(f"  HY: {len(hy_csv)} bytes")
    print(f"  IG: {len(ig_csv)} bytes")

    print("\nbuilding OAS reference CSVs...")
    try:
        hy_oas = _oas_history_csv("HY")
        ig_oas = _oas_history_csv("IG")
        print(f"  HY_OAS: {len(hy_oas)} bytes")
        print(f"  IG_OAS: {len(ig_oas)} bytes")
        has_oas = True
    except FileNotFoundError as exc:
        print(f"  skipping OAS CSVs: {exc}")
        has_oas = False

    print("\nbuilding trends.parquet (X matrix)...")
    trends_pq = _build_trends_parquet(x_columns)
    print(f"  trends.parquet: {len(trends_pq)/1024:.1f} KB")

    readme = _readme_text(bundle_version).encode("utf-8")

    print(f"\nwriting {out_path}...")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("HY_history.csv", hy_csv)
        z.writestr("IG_history.csv", ig_csv)
        z.writestr("trends.parquet", trends_pq)
        if has_oas:
            z.writestr("HY_OAS_history.csv", hy_oas)
            z.writestr("IG_OAS_history.csv", ig_oas)
        z.writestr("README.md", readme)
    print(f"  done — {out_path.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
