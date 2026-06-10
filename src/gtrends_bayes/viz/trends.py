"""Visualize raw vs. preprocessed Trends series."""

from __future__ import annotations

import pandas as pd


def plot_before_after(
    raw: pd.Series,
    processed: pd.Series,
    title: str | None = None,
):
    """Two-panel plot: raw SVI on top, post-pipeline series on bottom."""
    raise NotImplementedError("Phase 3+ — see IMPLEMENTATION_PLAN.md §3.")
