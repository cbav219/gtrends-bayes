"""AR(p) primary forecast + BSTS-on-residuals overlay.

Phase B's horizon sweep showed that AR(p) is already winning the IC race on
weekly cadence — beating BSTS-vs-AR head-to-head is a fool's errand. The
stacked-residual design takes the *right backbone* (AR(p)) and asks BSTS a
sharper question: *which Trends terms explain what AR(p) cannot?* That's a
smaller, denser signal — the dense-DGP regime Woloszko documents — and it's
where spike-and-slab inclusion probabilities become genuinely interpretable
because they're picking up cycle deviations rather than cycle level.

Hypothesis to test honestly: stacked RMSE < AR(p) RMSE on weekly horizon. If
it doesn't beat, document it as a null result; the inclusion-probability
output is still useful as a Trends Risk Index feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


class StackedResidualModel:
    """AR(p) primary + BSTS on residuals.

    Parameters
    ----------
    ar_p : int, default 4
        Lag order for the AR(p) backbone.
    bsts_kwargs : dict, optional
        Keyword args forwarded to ``BSTS()``. Defaults: ``n_seasons=52,
        expected_predictors=5, niter=1500, burn=150, seed=42``.
    """

    def __init__(self, ar_p: int = 4, bsts_kwargs: dict | None = None) -> None:
        self.ar_p = int(ar_p)
        self.bsts_kwargs = bsts_kwargs or {
            "n_seasons": 52,
            "expected_predictors": 5,
            "niter": 1500,
            "burn": 150,
            "seed": 42,
        }
        self._ar = None
        self._bsts = None
        self._y_history: pd.Series | None = None
        self._ar_resid: pd.Series | None = None
        self._fitted: bool = False

    # ---- core ---------------------------------------------------------------

    def fit(self, y: pd.Series, X: pd.DataFrame) -> StackedResidualModel:
        from gtrends_bayes.models.baseline import AR_p
        from gtrends_bayes.models.bsts import BSTS

        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be pandas.Series, got {type(y).__name__}")
        if X is None or X.shape[1] == 0:
            raise ValueError("X (Trends regressors) is required for the residual stage")
        if not y.index.equals(X.index):
            raise ValueError("y and X must share the same index")

        # Stage 1: AR(p) on y.
        self._ar = AR_p(p=self.ar_p, seed=42).fit(y)
        # In-sample AR residuals — length T - p, starting at index p.
        ar_resid_arr = np.asarray(self._ar._result.resid)
        ar_resid = pd.Series(
            ar_resid_arr,
            index=y.index[self.ar_p: self.ar_p + len(ar_resid_arr)],
            name="ar_residual",
        )
        self._ar_resid = ar_resid

        # Stage 2: BSTS on residuals.
        X_resid = X.loc[ar_resid.index]
        self._bsts = BSTS(**self.bsts_kwargs).fit(ar_resid, X_resid)

        self._y_history = y
        self._fitted = True
        log.info(
            "StackedResidualModel fit complete: AR(%d) on %d obs, BSTS on %d residuals",
            self.ar_p, len(y), len(ar_resid),
        )
        return self

    def forecast(
        self,
        horizon: int,
        X_future: pd.DataFrame,
        n_draws: int = 400,
    ) -> pd.DataFrame:
        """Combined forecast = AR(p) point forecast + BSTS posterior residual.

        Returns
        -------
        pandas.DataFrame
            ``(n_draws × horizon)``. Each row is a posterior path of the
            *combined* forecast. Use ``.attribution()`` for the in-sample
            decomposition.
        """
        self._require_fit()
        if X_future is None or X_future.shape[0] != horizon:
            raise ValueError(
                f"X_future must have {horizon} rows; got "
                f"{None if X_future is None else X_future.shape[0]}"
            )

        # AR forecast (point, deterministic — n_draws=1 then broadcast).
        ar_fc = self._ar.forecast(horizon=horizon, n_draws=1)
        ar_mean = np.asarray(ar_fc.mean(axis=0).values)  # (horizon,)

        # BSTS forecast of residuals. BSTS.forecast returns all post-burn draws
        # (its n_draws kwarg is currently inactive in the wrapper); subsample
        # to honor the requested n_draws.
        bsts_fc = self._bsts.forecast(horizon=horizon, X_future=X_future)
        bsts_arr = np.asarray(bsts_fc.values)
        if bsts_arr.shape[0] != n_draws:
            rng = np.random.default_rng(self.bsts_kwargs.get("seed", 42))
            idx = rng.choice(bsts_arr.shape[0], size=n_draws,
                             replace=(bsts_arr.shape[0] < n_draws))
            bsts_arr = bsts_arr[idx]

        # Combined: every BSTS draw + the deterministic AR mean per step.
        return pd.DataFrame(bsts_arr + ar_mean[np.newaxis, :])

    def attribution(self) -> pd.DataFrame:
        """In-sample decomposition: per-date split of AR vs Trends-residual contributions.

        Returns
        -------
        pandas.DataFrame indexed by date with columns:
            ``y``                    — observed
            ``ar_pred``              — AR(p) in-sample fitted value
            ``residual_pred``        — BSTS posterior median fit of the AR residual
            ``ar_share`` / ``trends_share`` — magnitude shares; sum to 1
        """
        self._require_fit()
        ar_fitted = pd.Series(
            np.asarray(self._ar._result.fittedvalues),
            index=self._y_history.index[self.ar_p: self.ar_p + len(self._ar._result.fittedvalues)],
            name="ar_pred",
        )
        # BSTS in-sample fit on residuals = sum of state-component medians.
        bands = self._bsts.component_bands()
        residual_pred = sum(b["q_med"] for b in bands.values())
        residual_pred.name = "residual_pred"

        common = ar_fitted.index.intersection(residual_pred.index)
        df = pd.DataFrame({
            "y": self._y_history.loc[common],
            "ar_pred": ar_fitted.loc[common],
            "residual_pred": residual_pred.loc[common],
        })
        df["total_pred"] = df["ar_pred"] + df["residual_pred"]
        denom = df["ar_pred"].abs() + df["residual_pred"].abs()
        df["ar_share"] = np.where(denom > 0, df["ar_pred"].abs() / denom, np.nan)
        df["trends_share"] = np.where(denom > 0, df["residual_pred"].abs() / denom, np.nan)
        return df

    def inclusion_probabilities(self) -> pd.Series:
        """Inclusion probabilities of the *residual* stage. Direct interpretability:
        these are the Trends predictors that explain what AR(p) misses."""
        self._require_fit()
        return self._bsts.inclusion_probabilities()

    def coefficient_summary(self) -> pd.DataFrame:
        self._require_fit()
        return self._bsts.coefficient_summary()

    # ---- housekeeping -------------------------------------------------------

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("StackedResidualModel has not been fit yet")

    def __repr__(self) -> str:
        return (
            f"StackedResidualModel(ar_p={self.ar_p}, "
            f"bsts_kwargs={self.bsts_kwargs}, fitted={self._fitted})"
        )
