"""Python wrapper around the R `bsts` package, called via rpy2.

Implementation strategy
-----------------------
Each ``BSTS`` instance gets a unique R-side model id. ``fit`` pushes ``y`` and
``X`` to R, calls ``fit_bsts(model_id, y, X, ...)``, and stores small posterior
summaries (coefficients, sigma, residuals, inclusion indicators) on the
Python side; the bulky R model object stays alive in ``.gtrends_models`` so
``predict`` can call ``predict.bsts`` natively without round-tripping the model.

Concurrency
-----------
``rpy2``'s R session is global within a Python process — you cannot truly
fit two BSTS instances in parallel inside one process. For the walk-forward
backtest in Phase 6 we use ``concurrent.futures.ProcessPoolExecutor`` with one
R subprocess per target, so the global-R limitation is harmless.
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gtrends_bayes.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# rpy2 bridge boot-strap. Lazy + module-global so import-time cost is zero
# until the first BSTS instance is constructed.
# ----------------------------------------------------------------------------

_R_INITIALIZED = False
_RO = None        # rpy2.robjects
_NUMPY2RI = None  # rpy2.robjects.numpy2ri
_PANDAS2RI = None  # rpy2.robjects.pandas2ri
_CONVERSION = None  # rpy2.robjects.conversion


def _init_r() -> None:
    """One-shot rpy2 setup: import R bridge, source the project's R helpers."""
    global _R_INITIALIZED, _RO, _NUMPY2RI, _PANDAS2RI, _CONVERSION
    if _R_INITIALIZED:
        return
    import rpy2.robjects as ro
    from rpy2.robjects import conversion, numpy2ri, pandas2ri

    _RO = ro
    _NUMPY2RI = numpy2ri
    _PANDAS2RI = pandas2ri
    _CONVERSION = conversion

    r_dir = Path(__file__).parent / "bsts_r"
    helpers = r_dir / "helpers.R"
    fit_script = r_dir / "fit_bsts.R"
    # Source helpers FIRST so fit_bsts.R can use them.
    ro.r(f'source("{helpers.as_posix()}")')
    ro.r(f'source("{fit_script.as_posix()}")')
    log.info("rpy2 bridge initialized; R bsts helpers loaded from %s", r_dir)
    _R_INITIALIZED = True


def _py2r_converter():
    """Composed converter that handles numpy + pandas in both directions."""
    return _CONVERSION.localconverter(
        _RO.default_converter + _NUMPY2RI.converter + _PANDAS2RI.converter
    )


def _as_str_list(x) -> list[str]:
    """Coerce an R character vector / numpy array / None to a list[str]."""
    if x is None or x is _RO.NULL:
        return []
    try:
        return [str(v) for v in x]
    except TypeError:
        return []


def _r_listvector_to_dict(lv) -> dict:
    """Turn an rpy2 named ListVector into a Python dict keyed by R names.

    rpy2's API for accessing element names varies across versions: in some
    builds ``lv.names`` is an attribute returning an R StrVector, in others
    it's a method. Fall back to the AttrPair API as a last resort.
    """
    names_attr = getattr(lv, "names", None)
    names_obj = names_attr() if callable(names_attr) else names_attr
    if names_obj is None or names_obj is _RO.NULL:
        # Try the attribute-pair fallback (pure-Python rpy2 path).
        try:
            names_obj = lv.do_slot("names")
        except Exception:  # noqa: BLE001
            return {names: lv[i] for i, names in enumerate(range(len(lv)))}
    names = [str(n) for n in names_obj]
    return {names[i]: lv[i] for i in range(len(lv))}


# ----------------------------------------------------------------------------
# BSTS class
# ----------------------------------------------------------------------------


class BSTS:
    """Bayesian Structural Time Series with spike-and-slab regression.

    Parameters
    ----------
    n_seasons : int, default 52
        ``AddSeasonal`` period. Set to 0 / None to disable seasonal component.
    expected_predictors : int, default 5
        Maps to R's ``expected.model.size`` — prior on number of included
        predictors under spike-and-slab.
    niter : int, default 3000
        MCMC iterations.
    burn : int or None
        Burn-in to discard on the Python side. Defaults to ``niter // 10``.
    seed : int, default 42
        Random seed (set on both Python and R sides).
    """

    def __init__(
        self,
        n_seasons: int = 52,
        expected_predictors: int = 5,
        niter: int = 3000,
        burn: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n_seasons = int(n_seasons)
        self.expected_predictors = int(expected_predictors)
        self.niter = int(niter)
        self.burn = int(burn) if burn is not None else self.niter // 10
        self.seed = int(seed)

        self._fitted: bool = False
        self._has_regression: bool = False
        self._model_id: str = f"bsts_{uuid.uuid4().hex[:8]}"
        self._predictor_names: list[str] = []
        # Includes "(Intercept)" if R's bsts added one (it always does for `y ~ .`).
        self._coefficient_names: list[str] = []
        self._target_index: pd.DatetimeIndex | None = None

        # Posterior summaries (post-burn unless noted).
        self._coefficients_post: np.ndarray | None = None      # (n_kept, p)
        self._inclusion_post: np.ndarray | None = None         # (n_kept, p) ints {0,1}
        self._sigma_obs_post: np.ndarray | None = None         # (n_kept,)
        self._one_step_residuals_post: np.ndarray | None = None  # (n_kept, T)
        self._state_component_names: list[str] = []

    # ---- core API -----------------------------------------------------------

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> BSTS:
        """Push ``y``, ``X`` to R, call ``fit_bsts``, store posterior summaries."""
        _init_r()
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas.Series, got {type(y).__name__}")
        self._target_index = y.index
        if X is not None and len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows; y has {len(y)}")

        y_arr = np.asarray(y.values, dtype=float)
        if X is None or X.shape[1] == 0:
            self._predictor_names = []
            X_for_r: Any = _RO.NULL
        else:
            self._predictor_names = list(X.columns)
            X_for_r = X.reset_index(drop=True)

        with _py2r_converter():
            result = _RO.r["fit_bsts"](
                model_id=self._model_id,
                y=y_arr,
                X=X_for_r,
                n_seasons=self.n_seasons,
                niter=self.niter,
                expected_model_size=self.expected_predictors,
                seed=self.seed,
            )
            res = _r_listvector_to_dict(result)
            self._has_regression = bool(res["has_regression"][0])
            sigma_obs = np.asarray(res["sigma_obs"])
            coefs = res["coefficients"]
            inclusion = res["inclusion_indicators"]
            residuals = res["one_step_residuals"]
            state_component_names = _as_str_list(res.get("state_contribution_names"))

        # Discard burn-in on our side (R's bsts does not).
        b = self.burn
        self._sigma_obs_post = sigma_obs[b:]
        if self._has_regression and coefs is not _RO.NULL:
            self._coefficients_post = np.asarray(coefs)[b:]
            self._inclusion_post = np.asarray(inclusion)[b:].astype(int)
            # R's `y ~ .` formula prepends "(Intercept)" — pick up the actual
            # coefficient names so subsequent indexing matches the matrix. R
            # wraps names with special chars in backticks (e.g. "`Credit &
            # Lending`") — strip them for clean plot labels.
            r_coef_names = [n.strip("`") for n in _as_str_list(res.get("coefficient_names"))]
            if r_coef_names:
                self._coefficient_names = r_coef_names
            else:
                self._coefficient_names = self._predictor_names
        if residuals is not _RO.NULL and residuals is not None:
            self._one_step_residuals_post = np.asarray(residuals)[b:]
        self._state_component_names = state_component_names
        self._fitted = True
        log.info(
            "BSTS fit done (model_id=%s, niter=%d, burn=%d, p=%d, has_regression=%s)",
            self._model_id, self.niter, self.burn,
            len(self._predictor_names), self._has_regression,
        )
        return self

    def inclusion_probabilities(self, include_intercept: bool = False) -> pd.Series:
        """Posterior P(γ_j = 1) per predictor (the key interpretability output).

        ``(Intercept)`` is hidden by default — bsts's spike-and-slab keeps it in
        the coefficient matrix but it's not a "predictor" in the project sense.
        """
        self._require_fit()
        if not self._has_regression:
            return pd.Series(dtype=float, name="inclusion_prob")
        probs = self._inclusion_post.mean(axis=0)
        s = pd.Series(probs, index=self._coefficient_names, name="inclusion_prob")
        if not include_intercept and "(Intercept)" in s.index:
            s = s.drop("(Intercept)")
        return s.sort_values(ascending=False)

    def coefficient_summary(self, include_intercept: bool = False) -> pd.DataFrame:
        """Per predictor: inclusion_prob, mean_when_included, sd_when_included, sign_consistency."""
        self._require_fit()
        if not self._has_regression:
            return pd.DataFrame(columns=["inclusion_prob", "mean_when_included",
                                         "sd_when_included", "sign_consistency"])
        coefs = self._coefficients_post
        incl = self._inclusion_post.astype(bool)
        rows = []
        for j, name in enumerate(self._coefficient_names):
            if not include_intercept and name == "(Intercept)":
                continue
            mask = incl[:, j]
            if not mask.any():
                rows.append((name, 0.0, np.nan, np.nan, np.nan))
                continue
            included = coefs[mask, j]
            mean_inc = float(included.mean())
            sd_inc = float(included.std(ddof=0))
            sign_consistency = float(np.mean(np.sign(included) == np.sign(mean_inc)))
            rows.append((name, float(mask.mean()), mean_inc, sd_inc, sign_consistency))
        df = pd.DataFrame(rows, columns=["predictor", "inclusion_prob",
                                          "mean_when_included", "sd_when_included",
                                          "sign_consistency"]).set_index("predictor")
        return df.sort_values("inclusion_prob", ascending=False)

    def forecast(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        n_draws: int | None = None,  # noqa: ARG002 — kept for API symmetry
    ) -> pd.DataFrame:
        """Posterior forecast paths via ``predict.bsts``.

        Returns
        -------
        pandas.DataFrame
            ``(n_draws_kept × horizon)``. Rows are MCMC draws (post-burn), columns
            are forecast steps 1..horizon.
        """
        self._require_fit()
        _init_r()
        if self._has_regression and X_future is None:
            raise ValueError("X_future is required because the fitted model has a regression component")
        if not self._has_regression and X_future is not None:
            log.warning("X_future provided but fitted model has no regression — ignoring")
            X_future = None

        with _py2r_converter():
            result = _RO.r["predict_bsts"](
                model_id=self._model_id,
                horizon=int(horizon if X_future is None else X_future.shape[0]),
                newdata=(X_future.reset_index(drop=True) if X_future is not None else _RO.NULL),
                burn=self.burn,
            )
            res = _r_listvector_to_dict(result)
            distribution = np.asarray(res["distribution"])  # (n_draws_kept, horizon)
        return pd.DataFrame(distribution)

    def component_bands(
        self,
        q_low: float = 0.05,
        q_high: float = 0.95,
    ) -> dict[str, pd.DataFrame]:
        """Posterior bands for each in-sample state component.

        Returns
        -------
        dict
            ``{component_name: DataFrame(date, q_low, q_med, q_high)}``.
            Component names come from ``model$state.contributions``'s 2nd dim.
        """
        self._require_fit()
        _init_r()
        with _py2r_converter():
            result = _RO.r["state_contributions"](model_id=self._model_id)
            res = _r_listvector_to_dict(result)
            arr = np.asarray(res["contributions"])  # (niter, n_state, T)
            names = _as_str_list(res.get("component_names"))
            if not names:
                names = [f"component_{i}" for i in range(arr.shape[1])]
        # Discard burn.
        arr = arr[self.burn:]
        out: dict[str, pd.DataFrame] = {}
        idx = self._target_index if self._target_index is not None else pd.RangeIndex(arr.shape[2])
        for k, name in enumerate(names):
            comp = arr[:, k, :]  # (n_draws, T)
            band = pd.DataFrame({
                "q_low": np.quantile(comp, q_low, axis=0),
                "q_med": np.quantile(comp, 0.5, axis=0),
                "q_high": np.quantile(comp, q_high, axis=0),
            }, index=idx)
            out[name] = band
        return out

    def to_arviz(self):
        """Build an arviz/xarray ``DataTree`` containing the posterior draws.

        ArviZ 1.0 retired ``InferenceData`` in favor of xarray's ``DataTree``
        (see the ArviZ migration guide). The returned object has a ``posterior``
        group with ``sigma_obs``, ``beta``, and ``inclusion`` variables ready
        for ``az.summary`` / ``az.plot_*`` calls.
        """
        self._require_fit()
        import xarray as xr

        # Conventionally: posterior dims (chain, draw, ...). bsts is single-chain.
        n_kept = self._sigma_obs_post.shape[0]
        data_vars: dict[str, tuple[list[str], np.ndarray]] = {
            "sigma_obs": (["chain", "draw"], self._sigma_obs_post[np.newaxis, :]),
        }
        coords: dict[str, list] = {"chain": [0], "draw": list(range(n_kept))}
        if self._has_regression and self._coefficients_post is not None:
            data_vars["beta"] = (
                ["chain", "draw", "coefficient"],
                self._coefficients_post[np.newaxis, :, :],
            )
            data_vars["inclusion"] = (
                ["chain", "draw", "coefficient"],
                self._inclusion_post[np.newaxis, :, :].astype(float),
            )
            coords["coefficient"] = self._coefficient_names
        posterior_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return xr.DataTree.from_dict({"posterior": posterior_ds})

    # ---- housekeeping -------------------------------------------------------

    def __del__(self) -> None:
        # Best-effort cleanup of the R-side model object on garbage collection.
        if self._fitted and _R_INITIALIZED and _RO is not None:
            with contextlib.suppress(Exception):
                _RO.r["delete_bsts"](self._model_id)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("BSTS instance has not been fit yet")

    def __repr__(self) -> str:
        return (
            f"BSTS(n_seasons={self.n_seasons}, expected_predictors={self.expected_predictors}, "
            f"niter={self.niter}, burn={self.burn}, seed={self.seed}, "
            f"fitted={self._fitted}, has_regression={self._has_regression})"
        )


def reset_r_models() -> None:
    """Tell R to discard every stored BSTS model. Call between walk-forward refits."""
    _init_r()
    _RO.r["delete_all_bsts"]()
