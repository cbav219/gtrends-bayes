"""Direction-of-change classifier — thin wrapper over a fitted BSTS posterior.

Computationally cheap: counts over existing forecast draws, no new MCMC.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class DirectionalForecaster:
    """Wraps a fitted BSTS to produce probabilities of direction / threshold events.

    Parameters
    ----------
    bsts : BSTS
        A fitted ``gtrends_bayes.models.bsts.BSTS`` instance.
    """

    def __init__(self, bsts) -> None:  # noqa: ANN001 — duck-typed
        if not getattr(bsts, "_fitted", False):
            raise ValueError("BSTS must be fit before wrapping in DirectionalForecaster")
        self.bsts = bsts

    def predict_proba(
        self,
        X_future: pd.DataFrame,
        y_baseline: float,
        n_draws: int = 1000,
        direction: Literal["increase", "decrease"] = "increase",
    ) -> pd.Series:
        """For each horizon step ``h``, P(forecast at h is on the chosen side of ``y_baseline``).

        Returns a ``Series`` indexed ``1..horizon``, values in ``[0, 1]``.

        Parameters
        ----------
        X_future : pandas.DataFrame
            Future regressor values; ``len(X_future)`` defines the horizon.
        y_baseline : float
            Reference level. Typically the most recently observed ``y``
            (i.e. the last in-sample observation).
        n_draws : int, default 1000
        direction : {"increase", "decrease"}, default "increase"
            ``"increase"`` returns ``P(y_h > y_baseline)`` — appropriate for
            OAS / yield-spread targets where positive Δ = widening.
            ``"decrease"`` returns ``P(y_h < y_baseline)`` — appropriate for
            ETF *price* targets where price-down = spread-up.
        """
        if direction not in ("increase", "decrease"):
            raise ValueError(f"unknown direction: {direction!r}")
        horizon = X_future.shape[0] if X_future is not None else 1
        paths = self.bsts.forecast(horizon=horizon, X_future=X_future, n_draws=n_draws)
        arr = np.asarray(paths.values)   # (n_draws_kept, horizon)
        if direction == "increase":
            probs = (arr > y_baseline).mean(axis=0)
        else:
            probs = (arr < y_baseline).mean(axis=0)
        s = pd.Series(probs, index=pd.RangeIndex(1, horizon + 1, name="horizon"),
                      name=f"p_{direction}")
        return s.clip(0.0, 1.0)

    def predict_proba_threshold(
        self,
        X_future: pd.DataFrame,
        y_baseline: float,
        threshold: float,
        n_draws: int = 1000,
        direction: Literal["above", "below", "either"] = "above",
    ) -> pd.Series:
        """For each horizon step, P(``y_h`` exceeds ``y_baseline`` by > ``threshold``).

        Use for the widening-event classifier (e.g. ``threshold=0.25`` on an
        ETF price = "drop by ≥ 25¢").

        Parameters
        ----------
        direction : {"above", "below", "either"}
            ``"above"``  → ``P(y_h > y_baseline + threshold)``
            ``"below"``  → ``P(y_h < y_baseline - threshold)``
            ``"either"`` → ``P(|y_h - y_baseline| > threshold)``
        """
        if direction not in ("above", "below", "either"):
            raise ValueError(f"unknown direction: {direction!r}")
        horizon = X_future.shape[0] if X_future is not None else 1
        paths = self.bsts.forecast(horizon=horizon, X_future=X_future, n_draws=n_draws)
        arr = np.asarray(paths.values)
        if direction == "above":
            probs = (arr > y_baseline + threshold).mean(axis=0)
        elif direction == "below":
            probs = (arr < y_baseline - threshold).mean(axis=0)
        else:
            probs = (np.abs(arr - y_baseline) > threshold).mean(axis=0)
        s = pd.Series(probs, index=pd.RangeIndex(1, horizon + 1, name="horizon"),
                      name=f"p_{direction}_t{threshold}")
        return s.clip(0.0, 1.0)
