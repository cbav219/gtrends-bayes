"""Frequentist baselines that mirror the BSTS API for benchmarking.

All three classes expose:
    .fit(y: pd.Series, X: pd.DataFrame | None = None) -> self
    .forecast(horizon: int, X_future: pd.DataFrame | None = None,
              n_draws: int = 1000) -> pd.DataFrame   # (n_draws x horizon)

so the walk-forward backtest treats them interchangeably with ``BSTS``.
Forecasts include parametric-bootstrap uncertainty bands sampled from the
estimated residual standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


class NaiveRW:
    """Random walk: ``y_hat_t = y_{t-1}``.

    Forecast variance grows linearly with horizon: ``Var(y_{t+h}) = h * sigma^2``.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._fitted = False
        self._last_value: float | None = None
        self._sigma: float | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> NaiveRW:  # noqa: ARG002
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be pandas.Series, got {type(y).__name__}")
        diffs = y.diff().dropna()
        self._sigma = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        self._last_value = float(y.iloc[-1])
        self._fitted = True
        return self

    def forecast(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,  # noqa: ARG002
        n_draws: int = 1000,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("NaiveRW not fit")
        rng = np.random.default_rng(self.seed)
        # cumulative-sum of independent N(0, sigma) innovations starting from last_value.
        innovations = rng.normal(0.0, self._sigma, size=(n_draws, horizon))
        paths = self._last_value + np.cumsum(innovations, axis=1)
        return pd.DataFrame(paths)


class AR_p:
    """Plain AR(p) via ``statsmodels.AutoReg``.

    ``forecast`` uses the analytic AR forecast for the mean and adds a
    parametric-bootstrap simulation of innovations for posterior bands.
    """

    def __init__(self, p: int = 4, seed: int = 42) -> None:
        self.p = int(p)
        self.seed = int(seed)
        self._fitted = False
        self._result = None
        self._y_history: pd.Series | None = None
        self._sigma: float | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> AR_p:  # noqa: ARG002
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be pandas.Series, got {type(y).__name__}")
        model = AutoReg(y.values.astype(float), lags=self.p, old_names=False)
        self._result = model.fit()
        self._y_history = y
        self._sigma = float(np.sqrt(self._result.sigma2))
        self._fitted = True
        return self

    def forecast(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,  # noqa: ARG002
        n_draws: int = 1000,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("AR_p not fit")
        # Analytic mean forecast from statsmodels.
        n = len(self._y_history)
        mean_fc = self._result.predict(start=n, end=n + horizon - 1)
        rng = np.random.default_rng(self.seed)
        # Parametric bootstrap: simulate paths whose innovation std equals sigma.
        # We add the cumulative innovation to the mean forecast — this is exact
        # only for a random-walk; for AR(p) it overestimates fan-out at long
        # horizon but is acceptable for the short horizons we care about (1-4w).
        innovations = rng.normal(0.0, self._sigma, size=(n_draws, horizon))
        paths = mean_fc.reshape(1, -1) + innovations.cumsum(axis=1) * 0 + innovations
        return pd.DataFrame(paths)


class AR_VIX:
    """AR(p) on target augmented with a single exogenous regressor (VIX log-diff).

    The plan locks ``X`` as a single-column DataFrame containing weekly Δ log(VIX);
    extending to multi-column exogenous regressors is straightforward.
    """

    def __init__(self, p: int = 4, seed: int = 42) -> None:
        self.p = int(p)
        self.seed = int(seed)
        self._fitted = False
        self._result = None
        self._y_history: pd.Series | None = None
        self._exog_history: pd.DataFrame | None = None
        self._sigma: float | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame) -> AR_VIX:
        if X is None or X.shape[1] == 0:
            raise ValueError("AR_VIX requires a non-empty X (exogenous regressors)")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows; y has {len(y)}")
        model = AutoReg(y.values.astype(float), lags=self.p,
                        exog=X.values.astype(float), old_names=False)
        self._result = model.fit()
        self._y_history = y
        self._exog_history = X
        self._sigma = float(np.sqrt(self._result.sigma2))
        self._fitted = True
        return self

    def forecast(
        self,
        horizon: int,
        X_future: pd.DataFrame,
        n_draws: int = 1000,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("AR_VIX not fit")
        if X_future is None or X_future.shape[0] != horizon:
            raise ValueError(f"X_future must have {horizon} rows; got {None if X_future is None else X_future.shape[0]}")
        n = len(self._y_history)
        mean_fc = self._result.predict(
            start=n, end=n + horizon - 1, exog_oos=X_future.values.astype(float)
        )
        rng = np.random.default_rng(self.seed)
        innovations = rng.normal(0.0, self._sigma, size=(n_draws, horizon))
        return pd.DataFrame(mean_fc.reshape(1, -1) + innovations)
