"""Notebook helpers — keep heavy logic out of the .ipynb cells.

Imported at the top of each notebook to set up paths, build the standard
Pipeline → feature-matrix flow, and build artifacts that are reused across
notebooks 04–07.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def project_root_from_cwd() -> Path:
    """Walk up from CWD until we find ``src/gtrends_bayes`` — set sys.path + cwd."""
    p = Path.cwd().resolve()
    while not (p / "src" / "gtrends_bayes").exists():
        if p == p.parent:
            raise RuntimeError("could not find src/gtrends_bayes/ above CWD")
        p = p.parent
    sys.path.insert(0, str(p / "src"))
    os.chdir(p)
    from dotenv import load_dotenv

    load_dotenv()
    return p


def build_full_feature_matrix(
    target_name: str,
    apply_target_transform: bool = False,
    nan_threshold: float = 0.5,
    include_controls: bool = True,
):
    """End-to-end: cache → Pipeline → feature matrix aligned to ``target_name``.

    Returns
    -------
    dict with keys: target_series, processed (DataFrame), pipeline, X, y,
    control_names (list[str]), pred_cfg, tgt_cfg.
    """
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
    processed_clean = drop_low_quality_columns(processed, nan_threshold=nan_threshold)

    target = load_target(target_name, tgt_cfg, apply_transform_field=apply_target_transform)
    X, y = build_feature_matrix(processed_clean, target, train_eligible=pipe.train_eligible_)

    control_names: list[str] = []
    if include_controls:
        controls = load_market_controls(tgt_cfg)
        # Align controls to X.index, drop rows where controls have NaN at the start.
        X, control_names = add_market_controls(X, controls)
        X = X.dropna()
        y = y.loc[X.index]

    return {
        "target_series": target,
        "processed": processed_clean,
        "pipeline": pipe,
        "X": X,
        "y": y,
        "control_names": control_names,
        "pred_cfg": pred_cfg,
        "tgt_cfg": tgt_cfg,
    }


def fit_bsts_default(y, X, niter: int = 1500, burn: int = 150,
                    expected_predictors: int = 5, seed: int = 42):
    """Fit BSTS with the project's standard hyperparameters."""
    from gtrends_bayes.models.bsts import BSTS

    model = BSTS(
        n_seasons=52,
        expected_predictors=expected_predictors,
        niter=niter,
        burn=burn,
        seed=seed,
    )
    model.fit(y, X)
    return model


def load_expected_predictors(target_name: str | None = None,
                              model_yaml: str = "config/model.yaml",
                              fallback: int = 5) -> int:
    """Look up the v2.1 per-target ``expected_model_size`` (Phase D.2 tuning).

    If ``config/model.yaml`` has
    ``bsts.prior.expected_model_size_per_target.{target_name}``, return that.
    Otherwise fall back to the legacy global ``bsts.prior.expected_model_size``
    or ``fallback``.
    """
    import yaml as _yaml
    from pathlib import Path

    p = Path(model_yaml)
    if not p.exists():
        return fallback
    raw = _yaml.safe_load(p.read_text()) or {}
    prior = raw.get("bsts", {}).get("prior", {})
    per_target = prior.get("expected_model_size_per_target") or {}
    if target_name and target_name in per_target:
        return int(per_target[target_name])
    return int(prior.get("expected_model_size", fallback))
