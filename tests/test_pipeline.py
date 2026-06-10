"""Tests for preprocessing.pipeline.Pipeline (Phase 3 orchestrator)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from gtrends_bayes.preprocessing.pipeline import Pipeline


@pytest.fixture
def long_panel(weekly_index, rng):
    """Long-form multi-sample fixture with 3 categories + 1 topic."""
    n = len(weekly_index)
    t = np.arange(n)
    drift = -0.025 * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / 52)

    rows = []
    for query, kind, intercept in [
        ("cat_60", "category", 60.0),
        ("cat_904", "category", 50.0),
        ("cat_958", "category", 45.0),
        ("/m/01jwbf", "topic", 30.0),
    ]:
        for sample_idx in range(3):
            svi = np.clip(intercept + drift + seasonal + rng.normal(0, 1.0, size=n), 1.0, 100.0)
            rows.append(pd.DataFrame({
                "date": weekly_index, "query": query, "sample_idx": sample_idx, "svi": svi,
            }))
    return pd.concat(rows, ignore_index=True)


def test_pipeline_runs_end_to_end(long_panel):
    pipe = Pipeline(classes={
        "cat_60": "category", "cat_904": "category", "cat_958": "category",
        "/m/01jwbf": "topic",
    })
    out = pipe.fit_transform(long_panel)
    assert isinstance(out, pd.DataFrame)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert set(out.columns) == {"cat_60", "cat_904", "cat_958", "/m/01jwbf"}
    assert pipe._fitted
    assert pipe.train_eligible_ is not None


def test_pipeline_train_mask_excludes_2011_and_2016(long_panel):
    pipe = Pipeline(classes={
        "cat_60": "category", "cat_904": "category", "cat_958": "category",
        "/m/01jwbf": "topic",
    })
    pipe.fit_transform(long_panel)
    excluded_years = pipe.train_eligible_.index[~pipe.train_eligible_].year.unique()
    assert set(excluded_years) == {2011, 2016}


def test_pipeline_preserves_topics_through_seasonality(long_panel):
    """Topic columns should NOT be YoY-differenced — they survive Phase-3 mostly intact
    after drift removal but in log-levels (no NaN run at the front)."""
    pipe = Pipeline(classes={
        "cat_60": "category", "cat_904": "category", "cat_958": "category",
        "/m/01jwbf": "topic",
    })
    out = pipe.fit_transform(long_panel)
    # Topic columns: relatively few NaN at the front (only what bias_removal might add).
    topic_nans = out["/m/01jwbf"].isna().sum()
    cat_nans = out["cat_60"].isna().sum()
    assert topic_nans < cat_nans, "categories should have ~52+ NaN from YoY diff"
    assert cat_nans >= 50  # YoY at 52 weeks introduces ≥52 NaN at the front


def test_pipeline_accepts_wide_input(long_panel):
    """If you've already averaged samples, the pipeline should accept the wide frame."""
    from gtrends_bayes.preprocessing.multi_sample import average_samples

    wide = average_samples(long_panel, drop_high_variance=False)
    pipe = Pipeline(classes={
        "cat_60": "category", "cat_904": "category", "cat_958": "category",
        "/m/01jwbf": "topic",
    })
    out = pipe.fit_transform(wide)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == wide.shape[1]


def test_pipeline_handles_empty_input():
    pipe = Pipeline()
    out = pipe.fit_transform(pd.DataFrame())
    assert out.empty
    assert pipe._fitted is True


def test_pipeline_is_pickleable(long_panel):
    """Required for walk-forward use across processes (one R subprocess per target)."""
    pipe = Pipeline(classes={"cat_60": "category"})
    pipe.fit_transform(long_panel)
    blob = pickle.dumps(pipe)
    revived: Pipeline = pickle.loads(blob)
    assert revived._fitted is True
    assert revived.classes == pipe.classes
    assert revived.train_eligible_.equals(pipe.train_eligible_)


def test_pipeline_rejects_non_datetime_wide_input():
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])  # int index
    with pytest.raises(ValueError, match="long-form .* or wide-form"):
        Pipeline().fit_transform(df)
