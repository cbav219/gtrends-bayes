"""Tests for features.groups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gtrends_bayes.config import PredictorsConfig
from gtrends_bayes.features.groups import (
    GROUP_DESCRIPTIONS,
    all_groups,
    group_columns,
    predictor_group,
    predictors_in_group,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def cfg():
    return PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")


def test_predictor_group_lookups_match_yaml(cfg):
    assert predictor_group("Jobs", cfg) == "labor"
    assert predictor_group("Bankruptcy", cfg) == "distress"
    assert predictor_group("Junk bond", cfg) == "credit_specific"
    assert predictor_group("Economic crisis", cfg) == "crisis"


def test_predictor_group_unknown_raises(cfg):
    with pytest.raises(KeyError):
        predictor_group("NotAThing", cfg)


def test_predictors_in_group(cfg):
    labor = predictors_in_group("labor", cfg)
    assert "Jobs" in labor and "Job Listings" in labor and "Recruitment & Staffing" in labor
    distress = predictors_in_group("distress", cfg)
    assert set(distress) == {"Bankruptcy", "Foreclosure"}


def test_group_columns_partitions_known_and_unmapped(cfg):
    cols = pd.Index(["Jobs", "Foreclosure", "vix", "ust10y_change"])
    parts = group_columns(cols, cfg)
    assert parts.get("labor") == ["Jobs"]
    assert parts.get("distress") == ["Foreclosure"]
    # Market controls land under the _unmapped bucket.
    assert set(parts.get("_unmapped", [])) == {"vix", "ust10y_change"}


def test_all_groups_distinct_and_described(cfg):
    groups = all_groups(cfg)
    # No duplicates.
    assert len(groups) == len(set(groups))
    # Every group key in the YAML must have a friendly description.
    for g in groups:
        assert g in GROUP_DESCRIPTIONS, f"missing GROUP_DESCRIPTIONS entry for {g!r}"
