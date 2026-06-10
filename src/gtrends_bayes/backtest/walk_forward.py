"""Pseudo-real-time walk-forward simulation with publication-lag discipline.

At simulation step ``t`` (counted from the start of the test window):
  - The model is trained on data with index ≤ ``t - publication_lag``.
  - The forecast target is ``y[t + h - 1]`` for each horizon ``h``.
  - For models with a regression component, ``X_future`` is taken as the most
    recently observed X — i.e. ``X[t - publication_lag]`` — extrapolated as a
    proxy for the unobserved future X (the standard nowcasting trick).
  - To keep BSTS runtime tractable, the model is re-fit only every
    ``refit_every`` steps; in between, the same fitted posterior is reused
    against fresh ``X_future``.

The simulator is model-agnostic: ``model_factory`` should return any object
with the project's ``fit(y, X)`` / ``forecast(horizon, X_future)`` interface
(``BSTS``, ``AR_p``, ``AR_VIX``, ``NaiveRW``).

Output formats
--------------
- **Single-horizon** (``horizons=[1]`` or legacy ``horizon=1``): one row per
  simulation step, indexed by the forecast date. Columns include
  ``y_true``, ``y_pred_mean``, the quantile bands, and ``refit``. Backward-
  compatible with v1 callers (notebook 07, existing tests).
- **Multi-horizon** (``horizons=[1, 2, 4, 8, 13]``): long format with a
  ``MultiIndex`` of (forecast_date, horizon). Same column set per row.

Phase D will add asymmetric publication-lag (``publication_lag_y`` /
``publication_lag_x``) and the ``mode = backtest | forecast`` distinction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    train_window: int = 260
    step: int = 1
    horizons: list[int] = field(default_factory=lambda: [1])
    refit_every: int = 13
    publication_lag: int = 1
    publication_lag_y: int = 0
    publication_lag_x: int = 1
    mode: Literal["backtest", "forecast"] = "backtest"


# Mode → default (publication_lag_y, publication_lag_x) when caller doesn't
# specify them explicitly. Backtest: we have everything (lag=0 for both).
# Forecast: y (ETF close) available same day, X (Trends) has ~5d delay.
_MODE_DEFAULTS: dict[str, tuple[int, int]] = {
    "backtest": (0, 0),
    "forecast": (0, 1),
}


class WalkForward:
    """Walk-forward backtester. See module docstring for the leakage rules."""

    def __init__(
        self,
        train_window: int = 260,
        step: int = 1,
        horizon: int | None = None,                 # legacy single-horizon
        horizons: list[int] | None = None,          # new multi-horizon
        refit_every: int = 13,
        publication_lag: int | None = None,         # legacy single-lag
        publication_lag_y: int | None = None,       # new asymmetric
        publication_lag_x: int | None = None,       # new asymmetric
        mode: Literal["backtest", "forecast"] = "backtest",
    ) -> None:
        # Reconcile the two horizon parameters.
        if horizons is not None and horizon is not None:
            raise ValueError("pass either horizon (legacy) or horizons (new), not both")
        if horizons is not None:
            resolved = list(int(h) for h in horizons)
        elif horizon is not None:
            resolved = [int(horizon)]
        else:
            resolved = [1]
        if not resolved:
            raise ValueError("horizons must have at least one entry")
        if any(h < 1 for h in resolved):
            raise ValueError(f"horizons must be >= 1; got {resolved}")

        # Reconcile lag parameters. Precedence:
        #   1. Explicit publication_lag_y / publication_lag_x take priority.
        #   2. Legacy publication_lag, if set, fills both.
        #   3. Mode defaults fill anything still unset.
        if mode not in _MODE_DEFAULTS:
            raise ValueError(f"mode must be 'backtest' or 'forecast'; got {mode!r}")
        default_y, default_x = _MODE_DEFAULTS[mode]
        if publication_lag is not None:
            if publication_lag_y is None:
                publication_lag_y = int(publication_lag)
            if publication_lag_x is None:
                publication_lag_x = int(publication_lag)
        if publication_lag_y is None:
            publication_lag_y = default_y
        if publication_lag_x is None:
            publication_lag_x = default_x
        if publication_lag_y < 0 or publication_lag_x < 0:
            raise ValueError("publication_lag_y / publication_lag_x must be >= 0")
        # Single effective lag = max of the two (the more conservative cutoff,
        # since training requires both X and y to be available).
        effective_lag = max(int(publication_lag_y), int(publication_lag_x))

        self.cfg = WalkForwardConfig(
            train_window=int(train_window),
            step=int(step),
            horizons=resolved,
            refit_every=int(refit_every),
            publication_lag=effective_lag,
            publication_lag_y=int(publication_lag_y),
            publication_lag_x=int(publication_lag_x),
            mode=mode,
        )

    # ---- core ---------------------------------------------------------------

    def run(
        self,
        model_factory: Callable[[], object],
        X: pd.DataFrame,
        y: pd.Series,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        n_draws: int = 1000,
    ) -> pd.DataFrame:
        """Simulate horizon-step-ahead forecasts.

        Parameters
        ----------
        model_factory : Callable[[], Model]
            Returns a fresh, unfit model. Called at every refit step.
        X : pandas.DataFrame
            Date-indexed feature matrix. Pass ``X.iloc[:, :0]`` (empty cols)
            for univariate models like ``NaiveRW`` and ``AR_p``.
        y : pandas.Series
            Date-indexed target. Must share the index of ``X``.
        start, end : Timestamp, optional
            Restrict the test window. Defaults: first date with enough
            history (``train_window + publication_lag`` after ``X.index[0]``);
            last date in ``X.index``.
        n_draws : int, default 1000
            Forwarded to model.forecast for posterior models that support it.

        Returns
        -------
        pandas.DataFrame
            Single-horizon: one row per forecast date with columns
            ``y_true, y_pred_mean, q025, ..., q975, refit``. Indexed by the
            forecast date.

            Multi-horizon: same columns, plus a ``horizon`` column. Indexed by
            ``(forecast_date, horizon)`` MultiIndex for easy ``.xs(h)``-style
            slicing.
        """
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same index")
        n = len(X)
        max_h = max(self.cfg.horizons)
        if n < self.cfg.train_window + self.cfg.publication_lag + max_h:
            raise ValueError(
                f"need at least {self.cfg.train_window + self.cfg.publication_lag + max_h} "
                f"observations; got {n}"
            )

        first_t = self.cfg.train_window + self.cfg.publication_lag
        if start is not None:
            first_t = max(first_t, X.index.searchsorted(start))
        # Need t + max_h - 1 < n so y[t + h - 1] exists for every horizon.
        last_t = n - max_h + 1
        if end is not None:
            last_t = min(last_t, X.index.searchsorted(end, side="right"))

        log.info(
            "WalkForward.run: T=%d, train_window=%d, lag=%d, refit_every=%d, "
            "horizons=%s, mode=%s, test indices [%d, %d) -> %d simulation steps",
            n, self.cfg.train_window, self.cfg.publication_lag,
            self.cfg.refit_every, self.cfg.horizons, self.cfg.mode,
            first_t, last_t,
            max(0, (last_t - first_t + self.cfg.step - 1) // self.cfg.step),
        )

        rows: list[tuple] = []
        current_model = None
        current_predictor_cols: list[str] = []
        steps_since_refit = 0
        wanted_quantiles = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
        q_cols = [f"q{int(round(q * 1000)):03d}" for q in wanted_quantiles]
        has_regression = X.shape[1] > 0
        single_horizon = len(self.cfg.horizons) == 1

        for t in range(first_t, last_t, self.cfg.step):
            # publication_lag semantics: pub_lag=N adds an N-step BUFFER between
            # the last training observation and the forecast target.
            #   pub_lag=0: train [.. t) exclusive (last row at t-1), predict y[t]
            #              -> straight 1-step-ahead, no artificial delay.
            #   pub_lag=1: train [.. t-1) exclusive (last row at t-2), predict y[t]
            #              -> simulates a 1-week Trends publication delay.
            train_end = t - self.cfg.publication_lag
            train_start = train_end - self.cfg.train_window
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]

            do_refit = (current_model is None) or (steps_since_refit >= self.cfg.refit_every)
            if do_refit:
                current_model = model_factory()
                if has_regression:
                    current_model.fit(y_train, X_train)
                else:
                    current_model.fit(y_train)
                current_predictor_cols = list(X_train.columns)
                steps_since_refit = 0
                log.debug("step t=%d: refit on %d obs ending %s",
                          t, len(y_train), str(y_train.index[-1].date()))
            steps_since_refit += 1

            # Forecast all horizons in one call. Repeat the most-recent observable
            # X row max_h times (nowcast: same X applies to every horizon step).
            # In asymmetric-lag mode, x_lookup_idx may differ from train_end-1
            # — we still use the most-recent OBSERVABLE X per the lag policy.
            if has_regression:
                # Latest observable X = X[t - pub_lag_x - 1]; clip to non-negative.
                x_lookup_idx = max(0, t - self.cfg.publication_lag_x - 1)
                latest_X_row = X.iloc[[x_lookup_idx]][current_predictor_cols]
                X_future = pd.concat([latest_X_row] * max_h, ignore_index=True)
                forecast_paths = current_model.forecast(
                    horizon=max_h, X_future=X_future, n_draws=n_draws,
                )
            else:
                forecast_paths = current_model.forecast(horizon=max_h, n_draws=n_draws)
            arr = np.asarray(forecast_paths.values)  # shape (n_draws, max_h)

            for h in self.cfg.horizons:
                target_idx = t + h - 1
                paths_at_h = arr[:, h - 1]
                row = {
                    "y_true": float(y.iloc[target_idx]),
                    "y_pred_mean": float(paths_at_h.mean()),
                    "refit": int(do_refit),
                }
                for q, col in zip(wanted_quantiles, q_cols):
                    row[col] = float(np.quantile(paths_at_h, q))
                rows.append((X.index[target_idx], h, row))

        if single_horizon:
            # Legacy wide format — index by forecast date only.
            idx = pd.DatetimeIndex([r[0] for r in rows], name="forecast_date")
            return pd.DataFrame([r[2] for r in rows], index=idx)

        # Multi-horizon long format.
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in rows], names=["forecast_date", "horizon"]
        )
        return pd.DataFrame([r[2] for r in rows], index=idx)


def collect_inclusion_history(
    backtest_results: pd.DataFrame,
    bsts_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    refit_every: int = 13,
    publication_lag: int = 1,
    train_window: int = 260,
) -> pd.DataFrame:  # noqa: ARG001 — placeholder for Phase 7 polish
    """Stub for tracking inclusion-prob churn across refits.

    Phase 7 will fill this in — for the v1 backtest table the per-refit
    inclusion histories aren't strictly needed.
    """
    raise NotImplementedError("Phase 7 polish — see IMPLEMENTATION_PLAN.md §3 Phase 7.")
