"""OECD economic-group taxonomy + credit-specific extensions.

Group membership for each predictor lives directly in
``config/predictors.yaml`` — the YAML's top-level keys under ``categories:``
and ``topics:`` ARE the group names. The helpers in this module read that
structure (via ``PredictorEntry.group``) and expose group ↔ predictor lookups
used downstream for:

- Shapley-style attribution post-fit (sum inclusion probabilities by group)
- v2 hierarchical priors with group-level shrinkage
- Side-by-side HY vs IG inclusion comparison plots
"""

from __future__ import annotations

import pandas as pd

from gtrends_bayes.config import PredictorsConfig

# Friendly descriptions for plot labels / docs. Keys MUST match the YAML group
# keys; if you add a new group to predictors.yaml, add a one-liner here too.
GROUP_DESCRIPTIONS: dict[str, str] = {
    "labor": "Jobs / unemployment-related categories",
    "credit_lending": "Consumer + commercial credit categories",
    "consumption": "Consumer-spending categories",
    "distress": "Bankruptcy + foreclosure categories",
    "industrial": "Industrial-activity categories",
    "finance_meta": "Generic finance categories (Investing, Insurance, ...)",
    "crisis": "Recession / crisis topics",
    "rates_and_macro": "Interest rates / Fed / inflation topics",
    "credit_specific": "Bond / mortgage / default topics — most directly OAS-relevant",
    "labor_topics": "Unemployment / layoff topics",
    "investment": "Investment / market topics",
}


def predictor_group(predictor_name: str, config: PredictorsConfig) -> str:
    """Return the YAML group key for a predictor referenced by its human name.

    Raises
    ------
    KeyError
        If ``predictor_name`` is not in ``config``.
    """
    for p in config.predictors:
        if p.name == predictor_name:
            return p.group
    raise KeyError(f"predictor {predictor_name!r} not found in config")


def predictors_in_group(group_name: str, config: PredictorsConfig) -> list[str]:
    """Return human names of every predictor whose group matches ``group_name``."""
    return [p.name for p in config.predictors if p.group == group_name]


def group_columns(
    columns: pd.Index | list[str],
    config: PredictorsConfig,
) -> dict[str, list[str]]:
    """Partition ``columns`` by predictor group.

    Columns whose name doesn't match any predictor in ``config`` are bucketed
    under the special key ``"_unmapped"`` (typically: market controls, lagged
    target features added later by the caller).
    """
    name_to_group = {p.name: p.group for p in config.predictors}
    out: dict[str, list[str]] = {}
    for col in columns:
        group = name_to_group.get(col, "_unmapped")
        out.setdefault(group, []).append(col)
    return out


def all_groups(config: PredictorsConfig) -> list[str]:
    """Distinct groups appearing in ``config``, in YAML declaration order."""
    seen: list[str] = []
    for p in config.predictors:
        if p.group not in seen:
            seen.append(p.group)
    return seen
