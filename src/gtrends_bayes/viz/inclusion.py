"""Visualize posterior inclusion probabilities (the key BSTS interpretability output)."""

from __future__ import annotations

import pandas as pd


def plot_inclusion(probs: pd.Series, top_k: int = 20, title: str | None = None):
    """Horizontal bar chart of top-k posterior inclusion probabilities."""
    raise NotImplementedError("Phase 5+ — see IMPLEMENTATION_PLAN.md §3.")


def plot_inclusion_compare(probs_a: pd.Series, probs_b: pd.Series, label_a: str, label_b: str):
    """Side-by-side bars for two targets (the HY vs IG comparison plot)."""
    raise NotImplementedError("Phase 6 — see IMPLEMENTATION_PLAN.md §3 Phase 6.")
