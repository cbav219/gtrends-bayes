"""Pydantic models that validate the YAML config files in ``config/``.

Loading helpers flatten the YAML's group-keyed structure into plain lists of
typed entries that are easier for downstream code to iterate.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

QueryKind = Literal["category", "topic", "keyword"]
Source = Literal["fred", "yfinance", "derived", "wrds", "parquet"]
Transform = Literal["levels", "log_diff", "diff"]


class PredictorEntry(BaseModel):
    """A single Trends predictor (category or topic)."""

    name: str
    kind: QueryKind
    group: str
    id: int | None = None        # for kind == "category"
    mid: str | None = None       # for kind == "topic"

    @field_validator("mid")
    @classmethod
    def _mid_only_for_topic(cls, v: str | None, info):  # noqa: ANN001
        if v is not None and info.data.get("kind") != "topic":
            raise ValueError("mid is only valid for topic predictors")
        return v


class SamplingConfig(BaseModel):
    n_samples: int = 6
    sleep_seconds: int = 60
    drop_high_variance: bool = True
    var_threshold: float = 25.0


class WindowConfig(BaseModel):
    start: date
    end: date
    train_end: date | None = None
    test_start: date | None = None


class PredictorsConfig(BaseModel):
    geo: str
    predictors: list[PredictorEntry]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    window: WindowConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PredictorsConfig":
        raw = yaml.safe_load(Path(path).read_text())
        flat: list[PredictorEntry] = []
        for group, entries in (raw.get("categories") or {}).items():
            for e in entries:
                flat.append(
                    PredictorEntry(name=e["name"], kind="category", group=group, id=e["id"])
                )
        for group, entries in (raw.get("topics") or {}).items():
            for e in entries:
                flat.append(
                    PredictorEntry(name=e["name"], kind="topic", group=group, mid=e["mid"])
                )
        return cls(
            geo=raw["geo"],
            predictors=flat,
            sampling=SamplingConfig(**(raw.get("sampling") or {})),
            window=WindowConfig(**raw["window"]),
        )


class TargetEntry(BaseModel):
    name: str
    description: str | None = None
    source: Source
    # ``ticker`` is required for FRED / yfinance / WRDS pulls. For
    # ``source: parquet`` (e.g. v5.1 HY_OAS / IG_OAS, built from the
    # FRED CSVs by oas_overlay_v3.py --write-targets), the data already
    # lives on disk and `path` points at the parquet file instead.
    ticker: str | None = None
    path: str | None = None
    units: str | None = None
    transform: Transform = "levels"
    # v3: the candidate transforms to fit and compare; the chosen winner is
    # written to `transform` and locked. None means "single fixed transform"
    # (the v1/v2 behavior).
    transforms: list[Transform] | None = None
    frequency: str = "weekly"
    week_anchor: str = "SUN"


class ControlEntry(BaseModel):
    name: str
    source: Source
    ticker: str | None = None
    formula: str | None = None
    transform: Transform = "levels"


class TargetsConfig(BaseModel):
    targets: list[TargetEntry]
    controls: list[ControlEntry] = Field(default_factory=list)
    window: WindowConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TargetsConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)


class BSTSStateSpec(BaseModel):
    local_linear_trend: bool = True
    seasonal: dict = Field(default_factory=lambda: {"enabled": True, "n_seasons": 52})


class BSTSPrior(BaseModel):
    expected_model_size: int = 5


class BSTSMcmc(BaseModel):
    niter: int = 3000
    burn_frac: float = 0.10
    seed: int = 42


class BSTSInference(BaseModel):
    forecast_draws: int = 1000
    forecast_horizon: int = 1


class BSTSConfig(BaseModel):
    state_spec: BSTSStateSpec = Field(default_factory=BSTSStateSpec)
    prior: BSTSPrior = Field(default_factory=BSTSPrior)
    mcmc: BSTSMcmc = Field(default_factory=BSTSMcmc)
    inference: BSTSInference = Field(default_factory=BSTSInference)


class BacktestConfig(BaseModel):
    train_window: int = 260
    step: int = 1
    horizon: int = 1
    refit_every: int = 13
    publication_lag: int = 1
    coverage_levels: list[float] = Field(default_factory=lambda: [0.5, 0.8, 0.95])


class PreprocessingCadenceConfig(BaseModel):
    hp_lambda: int
    periods_per_year: int
    yoy_diff_lag: int
    structural_break_drops_per_year: int = 1


class IngestConfig(BaseModel):
    """v3 ingest.yaml — daily-cadence and preprocessing-routing knobs.

    Loaded lazily; only required when ``cadence == "daily"`` paths are used.
    """
    daily_lag_business_days: int = 3
    preprocessing: dict[str, PreprocessingCadenceConfig] = Field(default_factory=dict)
    daily_chunker: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)


class ModelConfig(BaseModel):
    bsts: BSTSConfig = Field(default_factory=BSTSConfig)
    baselines: dict = Field(default_factory=dict)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)


# Backwards-compat shims (Phase 1 stubs).
def load_predictors(path: str | Path) -> PredictorsConfig:
    return PredictorsConfig.from_yaml(path)


def load_targets(path: str | Path) -> TargetsConfig:
    return TargetsConfig.from_yaml(path)


def load_model(path: str | Path) -> ModelConfig:
    return ModelConfig.from_yaml(path)
