"""CLI: convert a v3 BSTS posterior + AR backbone into a frozen v4 pickle.

Reads:
  - data/processed/posterior/{TARGET}_bsts_v1.pkl   (BSTS inclusion / coef summary, X_cols, y)
  - config/targets.yaml                              (target_transform)
  - config/model.yaml                                (BSTS state_spec, AR p)
  - data/processed/backtest/recalibration_alphas.json (conformal α, 0.80 level)

Writes:
  - dist/v4/model/{TARGET}_v4.pkl

Steps:
  1. Load v3 posterior pickle. Pull out inclusion_probs, coefficient_summary,
     X_columns, component_bands (stripped to mean+sd only), training y.
  2. Fit fresh AR(p) on the training y to extract the backbone (coefficients,
     intercept, sigma). p is read from config/model.yaml::stacked_residual.ar_p_per_target.
  3. Read conformal α from recalibration_alphas.json (BSTS-Trends row, 0.80 level).
  4. Build the preprocessing-state stub:
       drift_removal: identity (PCA = eye, mean = zeros)  ← see note
       yoy_periods_per_year: from config/ingest.yaml (or default 52 for weekly)
       structural_break_dates: [2011-01-01, 2016-01-01]
       cadence: from targets.yaml frequency
  5. Pickle the result.

**Note on preprocessing-state stub:** v2.1's preprocessing pipeline doesn't
save the learned HP-trend / PCA decomposition. v4's data-sideband contract
(IMPLEMENTATION_PLAN_v4.md §4.2) states the X parquet ships in preprocessed
form, so the inference module only does column-alignment defense. The
stored preprocessing-state fields are therefore metadata; identity PCA is
correct as long as the caller honors the data-sideband contract. If v3
Phase B's pipeline rewrites add real PCA persistence, this script picks it
up from the posterior pickle (key `preprocessing_state`) without changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gtrends_bayes.config import ModelConfig, TargetsConfig
from gtrends_bayes.logging import get_logger
from gtrends_bayes.models.baseline import AR_p
from gtrends_bayes.preprocessing.target_transform import TargetTransform

log = get_logger(__name__)

DEFAULT_POSTERIOR_DIR = Path("data/processed/posterior")
DEFAULT_OUT_DIR = Path("dist/v4/model")
DEFAULT_RECAL_PATH = Path("data/processed/backtest/recalibration_alphas.json")
DEFAULT_CONFORMAL_LEVEL = "0.8"   # 80% band — v4 USAGE.md primary report level
DEFAULT_CONFORMAL_MODEL_KEY = "BSTS (Trends)"
DEFAULT_OAS_OVERLAY_DIR = Path("data/processed/oas_overlay")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="freeze_model_v4")
    p.add_argument("--target", required=True,
                   help="Target name (e.g. 'HY', 'IG'). Must match targets.yaml.")
    p.add_argument("--posterior",
                   help="Override: explicit path to BSTS posterior pickle.")
    p.add_argument("--posterior-dir", default=str(DEFAULT_POSTERIOR_DIR))
    p.add_argument("--posterior-suffix", default="_bsts_v1.pkl",
                   help="Posterior filename = {target}{suffix}; default '_bsts_v1.pkl'.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--targets-config", default="config/targets.yaml")
    p.add_argument("--model-config", default="config/model.yaml")
    p.add_argument("--recal-path", default=str(DEFAULT_RECAL_PATH))
    p.add_argument("--conformal-level", default=DEFAULT_CONFORMAL_LEVEL,
                   help="Coverage level key in recalibration_alphas.json (default 0.8)")
    p.add_argument("--conformal-model-key", default=DEFAULT_CONFORMAL_MODEL_KEY)
    p.add_argument("--commit-hash", default=None,
                   help="v3 commit hash to embed (defaults to 'unknown')")
    p.add_argument("--bundle-version", default="v4",
                   help="Bundle version tag (e.g. 'v4', 'v5'). Affects output "
                        "filename suffix and the embedded metadata. Default 'v4'.")
    p.add_argument("--oas-overlay-dir", default=str(DEFAULT_OAS_OVERLAY_DIR),
                   help="Where the ETF↔OAS overlay artifacts live "
                        "(correlation.json + {target}_OAS_weekly.parquet). If "
                        "present *and* this target has overlay data, an "
                        "`oas_overlay_translation` block is embedded into the "
                        "frozen pickle so forecast() emits oas_implied_* fields.")
    return p


def _load_posterior(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"BSTS posterior pickle not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_ar_p(model_cfg: ModelConfig, target: str,
                  raw_model_yaml: dict) -> int:
    """Pull per-target AR-p from model.yaml::stacked_residual.ar_p_per_target.

    Falls back to baselines.ar_p.p (which is 4 in v2.1) if missing.
    """
    stacked = raw_model_yaml.get("stacked_residual") or {}
    per_target = stacked.get("ar_p_per_target") or {}
    if target in per_target:
        return int(per_target[target])
    baselines = model_cfg.baselines.get("ar_p", {})
    return int(baselines.get("p", 4))


def _detect_bsts_space(posterior: dict, declared_transform: str) -> str:
    """Auto-detect whether v2.1's BSTS was trained in level-space or transform-space.

    The v2.1 ETF posteriors were trained on raw levels even though
    ``targets.yaml`` declares ``transform: log_diff`` — that config field
    wasn't wired into the BSTS R bridge until v3 Phase C. We detect by
    comparing the in-sample fit's variance to the raw y's variance: if
    they're within a 2× factor, BSTS was trained on levels.

    Returns one of: ``"levels"``, ``"diff"``, ``"log_diff"``.
    """
    y = posterior.get("y")
    fit = posterior.get("in_sample_fit_median")
    if y is None or fit is None:
        # No diagnostic available; trust the declared transform.
        return declared_transform

    y_std = float(np.std(y.values))
    fit_std = float(np.std(fit.values))
    if y_std <= 0:
        return declared_transform

    ratio = fit_std / y_std
    # Levels-space BSTS: fit and y have similar variance (~0.5-2.0×).
    # Transform-space BSTS: fit variance is tiny relative to y (log-returns
    # are ~1-3% sd; y itself is dozens of USD or hundreds of bps).
    if 0.3 <= ratio <= 3.0:
        if declared_transform != "levels":
            log.warning(
                "Posterior y/fit variance ratio %.2f suggests BSTS was trained "
                "on LEVELS, but targets.yaml declares transform=%s. Overriding "
                "frozen target_transform → 'levels' so AR backbone fits the same "
                "space. (v3 Phase D retraining would fix this properly.)",
                ratio, declared_transform,
            )
        return "levels"
    return declared_transform


def _fit_ar_backbone(y_train: pd.Series, ar_p: int) -> dict:
    """Fit AR(p) on transformed y, return frozen-pickle ar_backbone dict."""
    if len(y_train) < ar_p + 5:
        raise RuntimeError(
            f"Not enough y history to fit AR({ar_p}): need {ar_p+5}, got {len(y_train)}"
        )
    ar = AR_p(p=ar_p)
    ar.fit(y_train)
    # statsmodels AR result exposes: .params (length p+1, intercept first by default)
    result = ar._result
    params = np.asarray(result.params).flatten()
    # AutoReg default with `old_names=False` returns [const, L1, L2, ...].
    intercept = float(params[0]) if len(params) == ar_p + 1 else 0.0
    if len(params) == ar_p + 1:
        coefs = params[1:]
    else:
        coefs = params[:ar_p]
    return {
        "p": int(ar_p),
        "coefficients": np.asarray(coefs, dtype=float),
        "intercept": float(intercept),
        "sigma": float(np.sqrt(result.sigma2)),
    }


def _resolve_conformal_alpha(
    recal: dict, target: str, model_key: str, level: str,
) -> float:
    """Look up the conformal multiplier for (target, model, level) from JSON."""
    if target not in recal:
        raise KeyError(f"recalibration_alphas.json missing target {target!r}; "
                       f"have {sorted(recal.keys())}")
    by_model = recal[target]
    if model_key not in by_model:
        raise KeyError(
            f"recalibration_alphas.json[{target!r}] missing model {model_key!r}; "
            f"have {sorted(by_model.keys())}"
        )
    by_level = by_model[model_key]
    if level not in by_level:
        raise KeyError(
            f"recalibration_alphas.json[{target!r}][{model_key!r}] "
            f"missing level {level!r}; have {sorted(by_level.keys())}"
        )
    return float(by_level[level]["alpha"])


def _canonicalize_coefficient_summary(
    cs: pd.DataFrame, x_cols: list[str],
) -> pd.DataFrame:
    """Rename v2.1's coefficient_summary columns to the v4 canonical schema.

    v2.1 emits ``inclusion_prob`` / ``mean_when_included`` / ``sd_when_included``
    / ``sign_consistency`` (indexed by predictor name, sorted by inclusion).
    v4 expects ``mean`` / ``sd`` / ``sign_consistency`` re-indexed by ``X_columns``
    so the inference module can use the same column order as the original X.
    """
    rename = {"mean_when_included": "mean", "sd_when_included": "sd"}
    out = cs.rename(columns=rename)
    # Re-index by X_columns; if a row is missing (shouldn't happen with v2.1
    # output), zero-fill so the Gaussian-mixture sampling is well-defined.
    out = out.reindex(x_cols).fillna({"mean": 0.0, "sd": 0.0, "sign_consistency": 0.5})
    # Drop the now-redundant inclusion_prob column (we keep the separate
    # inclusion_probs Series in bsts_posterior).
    out = out[[c for c in ("mean", "sd", "sign_consistency") if c in out.columns]]
    return out


def _build_oas_overlay_translation(
    target: str, overlay_dir: Path,
) -> dict | None:
    """Read the empirical ETF↔OAS overlay regression and pack as a dict.

    Returns ``None`` when the overlay artifacts aren't on disk or the target
    has no overlay row (e.g. the OAS-direct targets HY_OAS / IG_OAS, which
    forecast OAS bps directly and don't need a translation layer).

    The returned dict is embedded in the frozen pickle under
    ``oas_overlay_translation`` and consumed by
    :func:`gtrends_bayes.inference.forecast.forecast`, which emits
    ``oas_implied_median`` / ``oas_implied_band`` / ``oas_implied_path_*``
    alongside the level-space ETF forecast.

    Algebra: the OLS slope of ΔOAS-bps on ETF-Δlog over the 154-week overlap
    window is ``slope = pearson · σ_oas / σ_etf``. The implied OAS forecast
    at horizon h is ``oas_h = last_oas_bps + slope · ln(level_forecast_h /
    last_level)``. Pure linear-regression translation — no extra MCMC.
    """
    corr_path = overlay_dir / "correlation.json"
    parq_path = overlay_dir / f"{target}_OAS_weekly.parquet"
    if not corr_path.exists() or not parq_path.exists():
        return None
    corr = json.loads(corr_path.read_text())
    by_target = corr.get("by_target") or {}
    if target not in by_target:
        return None
    row = by_target[target]
    pearson = float(row["pearson"])
    sigma_etf = float(row["etf_dlog_std"])
    sigma_oas = float(row["oas_diff_std_bps"])
    if sigma_etf <= 0:
        return None
    # OLS slope of Δ-OAS-bps on ETF-Δlog (sign carries through pearson).
    slope = pearson * sigma_oas / sigma_etf

    # Latest weekly OAS level — needed as the anchor for the implied bps
    # forecast. The parquet has one column (HY_OAS_bps or IG_OAS_bps).
    oas = pd.read_parquet(parq_path)
    last_row_idx = oas.iloc[:, 0].dropna().index.max()
    last_oas_bps = float(oas.loc[last_row_idx].iloc[0])

    pearson_mag = abs(pearson)
    if pearson_mag >= 0.6:
        proxy_quality = "defensible"
    elif pearson_mag >= 0.4:
        proxy_quality = "moderate"
    else:
        proxy_quality = "weak"

    return {
        "slope_bps_per_dlog": round(slope, 4),
        "pearson": pearson,
        "spearman": float(row.get("spearman", float("nan"))),
        "n_overlap_weeks": int(row["n_obs"]),
        "overlap_start": str(row["overlap_start"]),
        "overlap_end": str(row["overlap_end"]),
        "last_oas_bps": last_oas_bps,
        "last_oas_date": pd.Timestamp(last_row_idx).date().isoformat(),
        "proxy_quality_label": proxy_quality,
        "source": "data/processed/oas_overlay (FRED BAML CSVs, 2023-05+)",
    }


def _strip_component_bands(component_bands: dict) -> dict:
    """Reduce per-component band arrays to mean + std summaries (saves bytes)."""
    out = {}
    for name, val in (component_bands or {}).items():
        if isinstance(val, (pd.DataFrame, pd.Series)):
            try:
                out[name] = {
                    "mean": float(np.asarray(val).mean()),
                    "std": float(np.asarray(val).std()),
                }
            except Exception:  # noqa: BLE001
                out[name] = None
        else:
            out[name] = None
    return out


def freeze(args: argparse.Namespace) -> Path:
    """Build a frozen v4 pickle from v3 artifacts; return its written path."""
    target = args.target
    posterior_path = (
        Path(args.posterior) if args.posterior
        else Path(args.posterior_dir) / f"{target}{args.posterior_suffix}"
    )
    log.info("freezing %s → %s", target, posterior_path)
    posterior = _load_posterior(posterior_path)

    # Sanity: pickle's `target` field should match the requested target.
    if posterior.get("target") != target:
        raise RuntimeError(
            f"Posterior pickle target={posterior.get('target')!r} doesn't "
            f"match requested --target {target!r}"
        )

    # 1. Load configs.
    targets_cfg = TargetsConfig.from_yaml(args.targets_config)
    target_entry = next((t for t in targets_cfg.targets if t.name == target), None)
    if target_entry is None:
        raise RuntimeError(
            f"target {target!r} not in {args.targets_config}; "
            f"have {[t.name for t in targets_cfg.targets]}"
        )
    declared_transform = target_entry.transform
    target_transform = _detect_bsts_space(posterior, declared_transform)
    cadence = "daily" if target_entry.frequency == "daily" else "weekly"
    yoy_periods = 252 if cadence == "daily" else 52

    model_cfg = ModelConfig.from_yaml(args.model_config)
    raw_model_yaml = yaml.safe_load(Path(args.model_config).read_text())
    ar_p = _resolve_ar_p(model_cfg, target, raw_model_yaml)

    # 2. Fit fresh AR(p) backbone on the *transformed* training y. The
    # inference module's _roll_ar_forward operates in transform-space, so
    # fitting on raw levels would explode the forecast under
    # TargetTransform.inverse_transform. log_diff: AR(p) on log-returns.
    # diff: AR(p) on bp-changes. levels: AR(p) on raw values.
    y_train_levels = posterior.get("y")
    if y_train_levels is None:
        raise RuntimeError(
            f"Posterior pickle missing 'y' (training series); cannot fit AR({ar_p})"
        )
    transform = TargetTransform(target_transform).fit(y_train_levels)
    y_train_transformed = transform.transform(y_train_levels).dropna()
    log.info(
        "fitting AR(%d) on transform=%s, n=%d obs",
        ar_p, target_transform, len(y_train_transformed),
    )
    ar_backbone = _fit_ar_backbone(y_train_transformed, ar_p)
    log.info(
        "AR(%d) fitted: intercept=%.5f, sigma=%.5f, coefs=%s",
        ar_p, ar_backbone["intercept"], ar_backbone["sigma"],
        np.array2string(ar_backbone["coefficients"], precision=4),
    )

    # 3. Conformal α from recalibration_alphas.json.
    recal_path = Path(args.recal_path)
    if recal_path.exists():
        recal = json.loads(recal_path.read_text())
        conformal_alpha = _resolve_conformal_alpha(
            recal, target, args.conformal_model_key, args.conformal_level,
        )
        log.info("conformal α (%s, %s, level=%s) = %.4f",
                 target, args.conformal_model_key, args.conformal_level,
                 conformal_alpha)
    else:
        log.warning("recalibration file %s not found; defaulting α=1.0", recal_path)
        conformal_alpha = 1.0

    # 4. Preprocessing-state stub. (See module docstring.)
    x_cols = list(posterior.get("X_columns") or [])
    k = len(x_cols)
    preprocessing_state = {
        "drift_removal": {
            "hp_lambda": int(model_cfg.bsts.state_spec.local_linear_trend) and 129_600,
            "pca_components": np.eye(k, dtype=float),
            "pca_mean": np.zeros(k, dtype=float),
        },
        "yoy_periods_per_year": yoy_periods,
        "structural_break_dates": [
            pd.Timestamp("2011-01-01"), pd.Timestamp("2016-01-01"),
        ],
        "cadence": cadence,
    }

    # 5. Build the frozen model dict.
    bsts_state_spec = {
        "local_linear_trend": bool(model_cfg.bsts.state_spec.local_linear_trend),
        "seasonal": dict(model_cfg.bsts.state_spec.seasonal),
    }
    # Re-index inclusion + coefficient summary by X_columns so the inference
    # module's `reindex(expected_cols)` is a no-op (preserving order).
    inclusion = posterior["inclusion_probs"].reindex(x_cols).fillna(0.0)
    coef_summary = _canonicalize_coefficient_summary(
        posterior["coefficient_summary"], x_cols,
    )

    # 5a. (Optional) embed the empirical ETF↔OAS regression so that
    # forecast() can emit oas_implied_* fields without the inference layer
    # needing to ship the FRED CSVs. Only populated for ETF targets (HY, IG)
    # — OAS-direct targets forecast bps natively and need no translation.
    oas_overlay_translation = _build_oas_overlay_translation(
        target, Path(args.oas_overlay_dir),
    )
    if oas_overlay_translation:
        log.info(
            "embedded OAS overlay: slope=%.2f bps/dlog, last_oas=%.0f bps "
            "(%s), pearson=%+.3f (%s)",
            oas_overlay_translation["slope_bps_per_dlog"],
            oas_overlay_translation["last_oas_bps"],
            oas_overlay_translation["last_oas_date"],
            oas_overlay_translation["pearson"],
            oas_overlay_translation["proxy_quality_label"],
        )

    # 5b. Self-describe which CSV in the data sideband holds this target's
    # history series. Lets verify_data.py and example_forecast.py pair each
    # pickle with the right history file without hardcoding a convention.
    history_file = (
        f"{target}_history.csv" if not target.endswith("_OAS")
        else f"{target}_history.csv"
    )

    frozen = {
        "target": target,
        "target_transform": target_transform,
        "bundle_version": args.bundle_version,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "v3_commit_hash": args.commit_hash or "unknown",
        "history_file": history_file,
        "ar_backbone": ar_backbone,
        "bsts_posterior": {
            "inclusion_probs": inclusion,
            "coefficient_summary": coef_summary,
            "state_spec": bsts_state_spec,
            "component_bands": _strip_component_bands(
                posterior.get("component_bands") or {}
            ),
            "X_columns": x_cols,
            # NOTE: MCMC draws + in_sample_fit_median + training y intentionally dropped.
        },
        "preprocessing": preprocessing_state,
        "conformal_alpha": conformal_alpha,
    }
    if oas_overlay_translation:
        frozen["oas_overlay_translation"] = oas_overlay_translation

    # 6. Write. Output filename suffix tracks --bundle-version so a v4 and
    # a v5 model can coexist side-by-side under the same out-dir during
    # the transition.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target}_{args.bundle_version}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(frozen, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = out_path.stat().st_size / 1024
    log.info("wrote %s (%.1f KB)", out_path, size_kb)
    if size_kb > 1536:
        log.warning("frozen pickle %s exceeds 1.5 MB target (got %.1f KB)",
                    out_path, size_kb)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out_path = freeze(args)
    except Exception as exc:  # noqa: BLE001
        log.error("freeze failed: %s", exc)
        raise
    log.info("freeze complete: %s", out_path)

    # Round-trip sanity check: load via inference.load_model and forecast on
    # the training y / synthetic X.
    from gtrends_bayes.inference import forecast, load_model
    model = load_model(out_path)
    log.info("round-trip load OK (target=%s, p=%d, conformal_α=%.3f)",
             model["target"], model["ar_backbone"]["p"],
             model["conformal_alpha"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
