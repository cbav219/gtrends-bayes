"""Tests for data.loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gtrends_bayes.config import PredictorsConfig
from gtrends_bayes.data.cache import cache_path, slugify, write_sample
from gtrends_bayes.data.loader import load_predictor_samples, predictor_classes

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_predictor_classes_human_names_keys():
    cfg = PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")
    classes = predictor_classes(cfg, rename_to_human=True)
    assert classes.get("Jobs") == "category"
    assert classes.get("Economic crisis") == "topic"
    assert all(v in ("category", "topic") for v in classes.values())


def test_predictor_classes_raw_identifier_keys():
    cfg = PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")
    classes = predictor_classes(cfg, rename_to_human=False)
    # "60" is the cat id for Jobs; the slugify mapping matches our cache layout.
    assert classes.get("60") == "category"
    # Find any topic mid currently in the YAML (mids change as Google
    # renumbers topics; don't pin to a literal value).
    sample_topic_mid = next(p.mid for p in cfg.predictors if p.kind == "topic")
    assert classes.get(sample_topic_mid) == "topic"


def test_load_predictor_samples_walks_cache(tmp_path: Path):
    """End-to-end: write a fake stitched sample into the cache layout, load it back."""
    cfg = PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")
    # Pick a predictor that exists in the config; fake one stitched sample.
    pred = next(p for p in cfg.predictors if p.name == "Jobs")
    slug = slugify(pred.id)  # Jobs is a category, uses id
    fake = pd.DataFrame({
        "date": pd.date_range("2020-01-05", periods=4, freq="W-SUN"),
        "query": [str(pred.id)] * 4,
        "sample_idx": [0] * 4,
        "svi": [50.0, 51.0, 49.0, 52.0],
    })
    p = cache_path(slug, cfg.geo, cfg.window.start, cfg.window.end, 0, root=tmp_path)
    write_sample(fake, p)

    long_df = load_predictor_samples(cfg, cache_root=tmp_path, rename_to_human=True)
    # query column was renamed to human name "Jobs".
    assert "Jobs" in long_df["query"].unique()
    assert len(long_df) == 4


def test_load_predictor_samples_empty_when_no_cache(tmp_path: Path):
    cfg = PredictorsConfig.from_yaml(CONFIG_DIR / "predictors.yaml")
    long_df = load_predictor_samples(cfg, cache_root=tmp_path)
    assert long_df.empty
    assert list(long_df.columns) == ["date", "query", "sample_idx", "svi"]
