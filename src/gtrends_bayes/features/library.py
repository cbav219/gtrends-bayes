"""Build the feature matrix that feeds both the HY and IG BSTS models.

The same ``X`` is returned for both fits — only ``y`` differs across the two
parallel models. The library also loads cached financial targets (HY, IG)
and macro controls (VIX, UST10Y change, 2y10y slope) from
``data/raw/targets/``, applies the per-column transforms specified in
``config/targets.yaml``, and exposes them in shapes BSTS can consume.
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from gtrends_bayes.config import TargetEntry, TargetsConfig
from gtrends_bayes.logging import get_logger

Transform = Literal["levels", "log_diff", "diff"]

DEFAULT_TARGETS_DIR = Path("data/raw/targets")
DEFAULT_BREAK_YEARS: tuple[int, ...] = (2011, 2016)
DEFAULT_CONTROLS: tuple[str, ...] = ("vix", "ust10y", "ust2y10y_slope")

log = get_logger(__name__)

_FORMULA_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9]+)\b")


# ---- Generic series helpers ------------------------------------------------

def apply_transform(series: pd.Series, transform: Transform) -> pd.Series:
    """Apply a per-series transform.

    Parameters
    ----------
    series : pandas.Series
    transform : {"levels", "log_diff", "diff"}
        - ``"levels"``: pass-through.
        - ``"log_diff"``: ``log(x_t) - log(x_{t-1})``. Series must be > 0.
        - ``"diff"``: first difference ``x_t - x_{t-1}``.
    """
    if transform == "levels":
        return series.copy()
    if transform == "log_diff":
        return np.log(series).diff()
    if transform == "diff":
        return series.diff()
    raise ValueError(f"unknown transform: {transform!r}")


# ---- Derived-formula evaluation (safe AST walk) ----------------------------

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv}


def _formula_dependencies(formula: str) -> list[str]:
    """Extract bare FRED-ticker tokens from a derived formula string."""
    return [t for t in _FORMULA_TICKER_RE.findall(formula) if t.isupper() and len(t) > 1]


def _safe_eval_formula(formula: str, lookup: dict[str, pd.Series]) -> pd.Series:
    """Evaluate ``formula`` using only +,-,*,/ over named Series in ``lookup``."""
    tree = ast.parse(formula, mode="eval").body

    def _eval(node):
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.Name):
            return lookup[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported AST node in formula {formula!r}: {ast.dump(node)}")

    return _eval(tree)


# ---- Target / control loading from data/raw/targets/ ----------------------

def _resolve_ticker_to_path(
    ticker: str,
    targets_config: TargetsConfig,
    target_dir: Path,
) -> Path:
    """Find the parquet file a FRED ticker was saved to (matches pull_targets layout)."""
    for c in targets_config.controls:
        if c.source == "fred" and c.ticker == ticker:
            return target_dir / f"{c.name}.parquet"
    for t in targets_config.targets:
        if t.source == "fred" and t.ticker == ticker:
            return target_dir / f"{t.name}.parquet"
    # Dependency-only ticker — pull_targets defaults to lowercase ticker as filename.
    candidate = target_dir / f"{ticker.lower()}.parquet"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no parquet found for FRED ticker {ticker!r} under {target_dir}")


def _read_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if df.shape[1] != 1:
        raise ValueError(f"expected single-column parquet at {path}, got {df.shape}")
    s = df.iloc[:, 0]
    s.index = pd.DatetimeIndex(s.index)
    return s


def load_target(
    name: str,
    targets_config: TargetsConfig,
    target_dir: Path = DEFAULT_TARGETS_DIR,
    apply_transform_field: bool = False,
) -> pd.Series:
    """Read a single target series (HY or IG) from disk.

    Parameters
    ----------
    name : str
        Target name as in ``targets_config.targets[].name`` (e.g. ``"HY"``).
    targets_config : TargetsConfig
    target_dir : Path
        Where ``pull_targets.py`` cached the parquets.
    apply_transform_field : bool, default False
        If True, apply the target's ``transform`` from the config (e.g. converts
        ETF prices to log-returns when ``transform: log_diff``). If False, the
        raw series is returned and downstream code (BSTS variant A vs B) decides.

    Returns
    -------
    pandas.Series
        Date-indexed, named ``name``.
    """
    target: TargetEntry | None = next((t for t in targets_config.targets if t.name == name), None)
    if target is None:
        raise KeyError(f"target {name!r} not in targets_config")
    path = target_dir / f"{name}.parquet"
    s = _read_series(path).rename(name)
    if apply_transform_field:
        s = apply_transform(s, target.transform)
    return s


def load_market_controls(
    targets_config: TargetsConfig,
    target_dir: Path = DEFAULT_TARGETS_DIR,
) -> dict[str, pd.Series]:
    """Load every control series, applying its declared transform.

    ``"derived"`` controls are computed from their FRED dependencies via
    ``_safe_eval_formula`` (a tiny AST walker that allows ``+ - * /`` over
    ticker names) so adding a new derived series only requires editing
    ``targets.yaml``.

    Returns
    -------
    dict
        ``{control_name: aligned_series}``. NaNs (e.g. the first row after a
        ``log_diff`` / ``diff``) are preserved — caller decides what to do
        about alignment.
    """
    out: dict[str, pd.Series] = {}
    for c in targets_config.controls:
        if c.source in ("fred", "yfinance") and c.ticker:
            path = target_dir / f"{c.name}.parquet"
            raw = _read_series(path)
            out[c.name] = apply_transform(raw, c.transform).rename(c.name)
        elif c.source == "derived" and c.formula:
            deps = _formula_dependencies(c.formula)
            lookup: dict[str, pd.Series] = {}
            for tkr in deps:
                lookup[tkr] = _read_series(_resolve_ticker_to_path(tkr, targets_config, target_dir))
            derived = _safe_eval_formula(c.formula, lookup)
            out[c.name] = apply_transform(derived, c.transform).rename(c.name)
        else:
            log.warning("skipping control %s: source=%s ticker=%s", c.name, c.source, c.ticker)
    return out


# ---- Feature matrix assembly ----------------------------------------------

def drop_low_quality_columns(
    X: pd.DataFrame,
    nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """Drop columns whose NaN fraction exceeds ``nan_threshold``.

    Use to prune predictors broken by chunk-stitching gaps (e.g. categories
    with empty middle chunks → near-constant log-epsilon → extreme YoY values
    → many NaN downstream).
    """
    if X.empty:
        return X.copy()
    nan_frac = X.isna().mean(axis=0)
    keep = nan_frac.index[nan_frac <= nan_threshold]
    dropped = sorted(set(X.columns) - set(keep))
    if dropped:
        log.warning("dropping %d low-quality columns (NaN frac > %.2f): %s",
                    len(dropped), nan_threshold, dropped)
    return X[keep].copy()


def build_feature_matrix(
    processed_df: pd.DataFrame,
    target: pd.Series,
    train_eligible: pd.Series | None = None,
    drop_break_years: tuple[int, ...] = DEFAULT_BREAK_YEARS,
    target_name: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align preprocessed Trends features with a target into (X, y).

    Drops rows where any column (in X or y) is NaN, then drops rows whose
    year is in ``drop_break_years``. If ``train_eligible`` is provided, it
    overrides the year-based exclusion (use the Pipeline's ``train_eligible_``
    mask for full consistency with preprocessing).

    Parameters
    ----------
    processed_df : pandas.DataFrame
        Output of ``Pipeline.fit_transform`` — wide, date-indexed.
    target : pandas.Series
        Date-indexed target (e.g. HY ETF price, possibly log-diffed).
    train_eligible : pandas.Series of bool, optional
    drop_break_years : tuple[int, ...], default (2011, 2016)
    target_name : str, optional
        Override the returned ``y.name`` if ``target`` is unnamed.

    Returns
    -------
    X : pandas.DataFrame  (date-indexed, no NaN, no break years)
    y : pandas.Series     (date-indexed, aligned with X)
    """
    if processed_df.empty or target.empty:
        return processed_df.copy(), target.copy()

    y_name = target_name or target.name or "y"
    y = target.rename(y_name)

    common = processed_df.index.intersection(y.index)
    if len(common) == 0:
        log.warning("no overlapping dates between processed_df and target")
        return processed_df.iloc[0:0].copy(), y.iloc[0:0].copy()

    joined = pd.concat([processed_df.loc[common], y.loc[common]], axis=1).dropna()

    if train_eligible is not None:
        eligible = train_eligible.reindex(joined.index, fill_value=True)
        joined = joined.loc[eligible]
    elif drop_break_years:
        joined = joined.loc[~joined.index.year.isin(drop_break_years)]

    X = joined.drop(columns=[y_name])
    y_out = joined[y_name]
    return X, y_out


def add_market_controls(
    X: pd.DataFrame,
    controls: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Augment ``X`` with macro control columns; return (X', control_names).

    Controls are concatenated as new columns and the join is left on ``X``'s
    index — out-of-range dates in the controls are dropped, missing rows in
    the controls become NaN. Caller is responsible for any further alignment
    / NaN handling (typically: rerun ``build_feature_matrix.dropna``-style
    cleanup at the end).

    The returned ``control_names`` list tells the BSTS wrapper which columns
    should have prior inclusion = 1.0 ("always-included").
    """
    if not controls:
        return X.copy(), []
    aligned = pd.DataFrame(dict(controls)).reindex(X.index)
    augmented = pd.concat([X, aligned], axis=1)
    return augmented, list(controls.keys())
