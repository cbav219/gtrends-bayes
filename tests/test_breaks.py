"""Tests for preprocessing.breaks (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gtrends_bayes.preprocessing.breaks import (
    EXCLUDED_YEARS,
    correct_jan_breaks,
)


def test_correct_jan_breaks_removes_injected_jump(synthetic_with_break):
    """Inject a +10 jump at 2011-01-01; correction should eliminate it.

    The fixture has noise ~N(0, 1). After correcting the +10 step, the
    post-correction series should be approximately flat (no level jump
    visible at the break boundary).
    """
    df = synthetic_with_break.to_frame(name="x")
    corrected, _ = correct_jan_breaks(df, break_dates=("2011-01-01",))
    # Average value before and after the break should now match within noise.
    pre = corrected["x"].loc[corrected.index < "2010-06-01"].mean()
    post = corrected["x"].loc[corrected.index >= "2011-06-01"].mean()
    assert abs(pre - post) < 1.5  # within ~1 sigma


def test_correct_jan_breaks_excludes_2011_and_2016(synthetic_with_break):
    df = synthetic_with_break.to_frame(name="x")
    _, train_eligible = correct_jan_breaks(df)
    excluded_count = (~train_eligible).sum()
    expected = ((df.index.year == 2011) | (df.index.year == 2016)).sum()
    assert excluded_count == expected
    assert set(train_eligible.index[~train_eligible].year.unique()) == set(EXCLUDED_YEARS)


def test_correct_jan_breaks_skips_out_of_range_breaks():
    """Break dates outside the data range should be ignored without error."""
    df = pd.DataFrame({"x": np.linspace(50, 60, 100)},
                      index=pd.date_range("2020-01-05", periods=100, freq="W-SUN"))
    out, _ = correct_jan_breaks(df, break_dates=("2011-01-01",))
    # No break inside the window → no correction → output equals input.
    pd.testing.assert_frame_equal(out, df)


def test_correct_jan_breaks_handles_empty_input():
    df = pd.DataFrame()
    out, mask = correct_jan_breaks(df)
    assert out.empty
    assert mask.empty


def test_correct_jan_breaks_preserves_columns_independently():
    """Two columns with different break magnitudes are corrected independently."""
    idx = pd.date_range("2009-01-04", "2013-12-29", freq="W-SUN")
    a = pd.Series(50.0, index=idx)
    b = pd.Series(50.0, index=idx)
    a.loc[idx >= "2011-01-01"] += 5.0
    b.loc[idx >= "2011-01-01"] += 15.0
    df = pd.DataFrame({"A": a, "B": b})
    corrected, _ = correct_jan_breaks(df, break_dates=("2011-01-01",))
    # Both columns flat after correction.
    assert abs(corrected["A"].loc[idx < "2010-06-01"].mean()
               - corrected["A"].loc[idx >= "2011-06-01"].mean()) < 1e-6
    assert abs(corrected["B"].loc[idx < "2010-06-01"].mean()
               - corrected["B"].loc[idx >= "2011-06-01"].mean()) < 1e-6
