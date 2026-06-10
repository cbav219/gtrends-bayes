"""Orchestrate the full OECD-Annex-A preprocessing chain in fit/transform form.

Order applied (per ``IMPLEMENTATION_PLAN.md`` §3 Phase 3 / Annex A of Paper 2):

    1. average_samples          (multi_sample.py)        long  -> wide
    2. log transform                                     wide  -> wide log-SVI
    3. remove_long_term_drift   (bias_removal.py)        wide  -> de-drifted log-SVI
    4. transform_by_class       (seasonality.py)         categories YoY-diffed,
                                                         topics passed through
    5. correct_jan_breaks       (breaks.py)              wide  -> wide + train mask

Phase-3 v1 limitation: ``fit`` and ``transform`` re-fit all stateful steps
(HP-trend + PCA) on the input each time — there is no true train-only state
re-use yet. This is fine for the walk-forward backtest pattern (call
``fit_transform`` at each refit step) but is NOT leakage-safe if you call
``fit(train)`` then ``transform(test)`` on a *strict* held-out window. The
proper separation lands when Phase 6 walk-forward is wired up.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger
from gtrends_bayes.preprocessing.bias_removal import remove_long_term_drift
from gtrends_bayes.preprocessing.breaks import DEFAULT_BREAKS, correct_jan_breaks
from gtrends_bayes.preprocessing.multi_sample import average_samples
from gtrends_bayes.preprocessing.seasonality import transform_by_class

QueryClass = Literal["category", "topic"]
Cadence = Literal["weekly", "daily"]

log = get_logger(__name__)

# Tiny offset added inside the log to make ``log(0)`` finite on series that
# briefly hit the SVI floor.
_LOG_EPSILON = 1e-3

# Cadence-specific defaults — only consulted when the caller doesn't pass an
# explicit override. Weekly values preserve the v1/v2 historical behavior;
# daily values follow IMPLEMENTATION_PLAN_v3.md §3.B.7.
_CADENCE_DEFAULTS = {
    "weekly": {"periods_per_year": 52},
    "daily": {"periods_per_year": 252},
}


class Pipeline:
    """Sequential preprocessing of multi-sample Trends pulls -> modeling-ready features.

    Parameters
    ----------
    classes : dict[str, {"category", "topic"}], optional
        Maps each predictor's column name to its query class. Columns not in
        ``classes`` default to ``"category"`` (i.e. they get YoY-differenced).
    hp_lambda : float, default 129_600
        Hodrick-Prescott smoothing parameter inside ``remove_long_term_drift``.
        129600 is the canonical weekly default.
    weighted_neighbor : bool, default True
        Use the (0.25, 0.5, 0.25) weighted prior-year reference in the YoY
        log-diff (the OECD weekly-tracker variant).
    break_dates : iterable of str, default ("2011-01-01", "2016-01-01")
    var_threshold : float, default 25.0
        Drop multi-sample series whose mean cross-sample std exceeds this.
    drop_high_variance : bool, default True
    apply_breaks : bool, default True
        If False, ``correct_jan_breaks`` is skipped (still computes the
        training-eligible mask).
    """

    def __init__(
        self,
        classes: dict[str, QueryClass] | None = None,
        hp_lambda: float = 129_600.0,
        weighted_neighbor: bool = True,
        break_dates: Iterable[str] = DEFAULT_BREAKS,
        var_threshold: float = 25.0,
        drop_high_variance: bool = True,
        apply_breaks: bool = True,
        cadence: Cadence = "weekly",
        periods_per_year: int | None = None,
    ) -> None:
        self.classes = dict(classes or {})
        self.hp_lambda = float(hp_lambda)
        self.weighted_neighbor = bool(weighted_neighbor)
        self.break_dates = tuple(break_dates)
        self.var_threshold = float(var_threshold)
        self.drop_high_variance = bool(drop_high_variance)
        self.apply_breaks = bool(apply_breaks)
        if cadence not in _CADENCE_DEFAULTS:
            raise ValueError(f"cadence={cadence!r} not in {tuple(_CADENCE_DEFAULTS)}")
        self.cadence: Cadence = cadence
        self.periods_per_year = int(
            periods_per_year
            if periods_per_year is not None
            else _CADENCE_DEFAULTS[cadence]["periods_per_year"]
        )
        # Populated by fit / fit_transform.
        self.train_eligible_: pd.Series | None = None
        self._fitted: bool = False

    # ---- Stateful API -------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Pipeline:
        """Run the pipeline on ``df`` to populate ``train_eligible_``.

        Currently equivalent to ``fit_transform`` but discards the returned
        frame. Provided for sklearn-style symmetry; future versions will store
        learned PCA components for true train/test separation.
        """
        self.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the pipeline to ``df``.

        Phase 3 v1: re-fits stateful steps each call (no leakage protection).
        """
        return self.fit_transform(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """End-to-end preprocessing in one shot. Returns the wide processed frame."""
        if df.empty:
            self.train_eligible_ = pd.Series(dtype=bool)
            self._fitted = True
            return df.copy()

        # 1. Multi-sample averaging (only if input is long-form with sample_idx).
        if {"date", "query", "sample_idx", "svi"}.issubset(df.columns):
            log.debug("Pipeline step 1: averaging multi-sample draws")
            wide = average_samples(
                df,
                drop_high_variance=self.drop_high_variance,
                var_threshold=self.var_threshold,
            )
        elif isinstance(df.index, pd.DatetimeIndex):
            log.debug("Pipeline step 1: input already wide and date-indexed; skipping average")
            wide = df.copy()
        else:
            raise ValueError(
                "Pipeline expects either long-form (date|query|sample_idx|svi) or"
                " wide-form date-indexed input"
            )

        if wide.empty:
            self.train_eligible_ = pd.Series(dtype=bool)
            self._fitted = True
            return wide

        # 2. log transform.
        log.debug("Pipeline step 2: log transform")
        log_svi = np.log(wide.clip(lower=_LOG_EPSILON))

        # 3. Long-term drift removal.
        log.debug("Pipeline step 3: remove_long_term_drift (lambda=%.0f)", self.hp_lambda)
        de_drifted = remove_long_term_drift(log_svi, hp_lambda=self.hp_lambda)

        # 4. Class-aware seasonality.
        log.debug(
            "Pipeline step 4: transform_by_class (cadence=%s, periods_per_year=%d)",
            self.cadence, self.periods_per_year,
        )
        seasonal = transform_by_class(
            de_drifted,
            classes=self.classes,
            periods_per_year=self.periods_per_year,
            weighted_neighbor=self.weighted_neighbor,
        )

        # 5. January-break corrections + training-eligible mask.
        log.debug("Pipeline step 5: correct_jan_breaks")
        if self.apply_breaks:
            corrected, train_mask = correct_jan_breaks(seasonal, break_dates=self.break_dates)
        else:
            corrected = seasonal
            from gtrends_bayes.preprocessing.breaks import EXCLUDED_YEARS

            train_mask = pd.Series(
                ~seasonal.index.year.isin(EXCLUDED_YEARS),
                index=seasonal.index,
                name="train_eligible",
            )

        self.train_eligible_ = train_mask
        self._fitted = True
        return corrected

    # ---- Conveniences -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Pipeline(classes={len(self.classes)} entries, "
            f"hp_lambda={self.hp_lambda}, weighted_neighbor={self.weighted_neighbor}, "
            f"break_dates={self.break_dates}, var_threshold={self.var_threshold}, "
            f"cadence={self.cadence}, periods_per_year={self.periods_per_year}, "
            f"fitted={self._fitted})"
        )
