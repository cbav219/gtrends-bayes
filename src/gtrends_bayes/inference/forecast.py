"""Forecast horizon-step-ahead level + 90% credible band from a frozen model.

This file contains the core ``forecast()`` function — the *entry point*
the PM calls directly (or via :mod:`gtrends_bayes.inference.cli`). It is
the file most likely to be opened by a PM auditing how the model works.

The 7-step algorithm
--------------------
For a given ``(model, horizon, as_of, y_history, x_latest)``:

1. **Validate inputs** — confirm y has enough history for the AR(p)
   backbone and x carries all expected predictor columns.
2. **Transform y** — apply the model's ``target_transform`` (``levels``,
   ``diff``, or ``log_diff``) so the AR state aligns with how the model
   was fit. Sets ``last_state`` = the most recent ``p`` transformed
   observations.
3. **AR(p) mean forecast** — analytic recursion using the frozen
   coefficients. No noise yet, just the conditional mean.
4. **BSTS-residual sampling** — draw ``n_draws`` posterior βs via the
   spike-and-slab approximation
   ``β_k ~ Bernoulli(inclusion_prob_k) · 𝒩(mean_k, sd_k)``,
   then multiply by the latest X row (nowcasting trick: assume the X
   signal stays at its latest observed value over the horizon).
5. **Add cumulative Gaussian state noise** — innovation σ from the AR
   backbone, summed over the path so variance grows ~√t with horizon.
6. **Apply conformal multiplier α** — inflates (or sometimes shrinks)
   the 90% band so its empirical coverage matches nominal on the
   validation slice. Per-target α is stored in the pickle.
7. **Inverse-transform to level space** — re-aggregate the
   transform-space forecasts back to the target's native units (USD for
   ETF prices, bps for OAS) for PM consumption.

Output is a dict with both the transform-space numbers (``median``,
``q05``, ``q95``) and the level-space numbers (``level_median``,
``level_band``), plus full forecast paths for plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gtrends_bayes.preprocessing.target_transform import TargetTransform

from .preprocess import apply_preprocessing

# Map human-friendly horizon labels to business-day counts. Same ladder
# as IMPLEMENTATION_PLAN_v3.md §6.E.1. PMs pass strings; ``forecast()``
# accepts either a label here or an explicit int business-day count.
HORIZON_BD: dict[str, int] = {
    "1d": 1, "1w": 5, "2w": 10, "1m": 21, "1q": 63, "6m": 126, "1y": 252,
}


def _resolve_horizon(horizon: str | int) -> tuple[int, str]:
    """Return ``(business_days, label)`` for a string or int horizon.

    Raises ``ValueError`` if a string isn't in :data:`HORIZON_BD` or an
    int is non-positive.
    """
    if isinstance(horizon, str):
        if horizon not in HORIZON_BD:
            raise ValueError(
                f"horizon={horizon!r} not in {sorted(HORIZON_BD.keys())}"
            )
        return HORIZON_BD[horizon], horizon
    h = int(horizon)
    if h <= 0:
        raise ValueError(f"horizon must be positive; got {h}")
    return h, f"{h}bd"


def _roll_ar_forward(
    history_state: np.ndarray,
    phi: np.ndarray,
    intercept: float,
    horizon: int,
) -> np.ndarray:
    """Analytic AR(p) conditional-mean forecast over ``horizon`` steps.

    Implements the deterministic recursion
    ``y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + ... + φ_p·y_{t-p}``,
    no innovation noise. The noise is added separately in step 5 so we
    can sample it ``n_draws`` times in vectorized form.

    Parameters
    ----------
    history_state : ndarray
        Last ``p`` observed values of y (transform-space), oldest-first.
    phi : ndarray
        AR coefficient vector of length ``p``.
    intercept : float
        AR intercept ``c``.
    horizon : int
        Number of steps to roll forward.

    Returns
    -------
    ndarray of shape ``(horizon,)``
        Conditional-mean forecast at each step ``1..horizon``.
    """
    p = len(phi)
    state = history_state[-p:].astype(float).copy()
    out = np.empty(horizon, dtype=float)
    for t in range(horizon):
        # AR convention: y_t = c + phi[0] * y_{t-1} + phi[1] * y_{t-2} + ...
        # `state` is stored oldest-first, so reverse it to align with phi.
        y_next = intercept + np.dot(phi, state[::-1])
        out[t] = y_next
        # Slide the window forward: drop oldest, append the just-forecast value.
        state = np.concatenate([state[1:], [y_next]])
    return out


def forecast(
    model: dict[str, Any],
    horizon: str | int,
    as_of: pd.Timestamp,
    y_history: pd.Series,
    x_latest: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single horizon-step-ahead forecast from a frozen model.

    Parameters
    ----------
    model : dict
        Loaded via :func:`gtrends_bayes.inference.load_model`. Schema
        documented in ``IMPLEMENTATION_PLAN_v5.md`` and in
        :mod:`gtrends_bayes.inference.load`.
    horizon : str or int
        One of ``"1d", "1w", "2w", "1m", "1q", "6m", "1y"``, or an
        explicit integer number of business days.
    as_of : pandas.Timestamp
        The "decision day" — the timestamp the PM is forecasting *from*.
        All input data must cover history up to (or before) this date.
    y_history : pandas.Series
        Recent target levels (HYG / LQD ETF closing price in USD for
        v5). Must contain at least ``model["ar_backbone"]["p"]``
        observations *after* the model's target_transform is applied.
    x_latest : pandas.DataFrame
        Recent preprocessed Trends matrix. Columns must include every
        name in ``model["bsts_posterior"]["X_columns"]``; index = dates.
    n_draws : int, default 1000
        Number of posterior draws used to estimate the credible band.
        Reduce to ~200 for faster live monitoring; increase for tighter
        Monte-Carlo error.
    seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        ``target``, ``target_transform``, ``as_of``, ``horizon``,
        ``horizon_bd``, ``n_draws``, ``conformal_alpha``
            Echo of the inputs + model metadata.

        ``median``, ``q05``, ``q95``
            Point forecast + 90% band at the terminal step (h),
            in **transform space** (e.g. log-returns if
            ``target_transform="log_diff"``).

        ``level_median``, ``level_band``
            Same scalars re-aggregated to **level space** (USD for ETF
            prices, bps for OAS). ``level_band`` is a 2-tuple
            ``(q05_level, q95_level)``.

        ``path_median``, ``path_q05``, ``path_q95``
            Full transform-space paths of length ``horizon_bd`` —
            useful for fan-chart plotting.

        ``level_path_median``, ``level_path_q05``, ``level_path_q95``
            Same paths in level space.

    Examples
    --------
    >>> model = load_model("model/HY_v5.pkl")
    >>> out = forecast(model, "1m", pd.Timestamp("2026-05-03"), y, x)
    >>> out["level_median"]      # e.g. 80.56
    >>> out["level_band"]        # e.g. (72.42, 88.87)
    """
    # ---- 1. Validate + resolve horizon -----------------------------------
    h, h_label = _resolve_horizon(horizon)
    as_of_ts = pd.Timestamp(as_of)
    rng = np.random.default_rng(int(seed))

    bsts = model["bsts_posterior"]
    expected_cols: list[str] = list(bsts["X_columns"])

    # Align X to the model's canonical column order. Raises a clear
    # error if any predictor is missing. See preprocess.py for the
    # reason this matters (matrix-multiply alignment).
    x_aligned = apply_preprocessing(
        x_latest, model["preprocessing"], expected_columns=expected_cols,
    )

    # ---- 2. Transform y_history into the model's working space ----------
    # For 'log_diff', this converts ETF prices into weekly log-returns —
    # the space the AR(p) backbone was fit on. For 'levels' (v5 default
    # because build_features() didn't apply the transform), this is a
    # no-op and AR runs directly on prices. Either way, last_state is
    # the most recent p transformed values.
    transform = TargetTransform(model["target_transform"])
    transform.fit(y_history)
    y_transformed = transform.transform(y_history).dropna()

    p = int(model["ar_backbone"]["p"])
    if len(y_transformed) < p:
        raise ValueError(
            f"y_history has {len(y_transformed)} obs after transform; "
            f"AR({p}) needs at least {p}"
        )
    last_state = y_transformed.iloc[-p:].values.astype(float)

    # ---- 3. AR(p) conditional-mean forecast over h steps ----------------
    # Pure deterministic recursion using the frozen AR coefficients.
    # Innovation noise is added in step 5 to get the credible band.
    phi = np.asarray(model["ar_backbone"]["coefficients"], dtype=float).flatten()
    if len(phi) != p:
        raise ValueError(
            f"ar_backbone.coefficients length {len(phi)} != p={p}"
        )
    intercept = float(model["ar_backbone"]["intercept"])
    ar_path_mean = _roll_ar_forward(last_state, phi, intercept, h)

    # ---- 4. BSTS-residual sampling via Gaussian-mixture approximation ---
    # Spike-and-slab posterior: each predictor k has
    #   P(included)  = inclusion_probs[k]
    #   β_k | included ~ 𝒩(coefficient_summary["mean"][k],
    #                       coefficient_summary["sd"][k])
    # We sample β = Bernoulli(P_incl) · 𝒩(mean, sd) per draw, then
    # contract with the latest X row to get the BSTS contribution at
    # each draw.
    coef_summary = bsts["coefficient_summary"]
    inclusion = bsts["inclusion_probs"]

    means = (
        coef_summary["mean"].reindex(expected_cols).fillna(0.0).values
    )
    sds = (
        coef_summary["sd"].reindex(expected_cols).fillna(0.0).values
    )
    incl_p = (
        inclusion.reindex(expected_cols).fillna(0.0).values
    )

    k = len(expected_cols)
    # n_draws × k draws of inclusion masks (0/1) and normal coefficients.
    incl_draws = rng.binomial(1, np.clip(incl_p, 0.0, 1.0), size=(n_draws, k))
    coef_draws = rng.normal(means, np.maximum(sds, 1e-12), size=(n_draws, k))
    # Element-wise zeroing of excluded coefficients gives the effective β.
    betas = coef_draws * incl_draws  # n_draws × k

    # Nowcasting assumption: the X signal stays at its latest observed
    # value over the forecast horizon. (Trends don't have a usable
    # forward forecast of their own.) So we use x_now repeated h times.
    x_now = x_aligned.iloc[-1:].values.astype(float)  # 1 × k
    bsts_path_means = betas @ x_now.T                  # n_draws × 1
    bsts_paths = np.broadcast_to(bsts_path_means, (n_draws, h)).copy()

    # ---- 5. Cumulative Gaussian state noise -----------------------------
    # AR innovation σ is the per-step state noise. We accumulate it over
    # the horizon so variance grows linearly in horizon (√h on the std),
    # which is the correct asymptotic for diff/log_diff space.
    sigma = float(model["ar_backbone"]["sigma"])
    state_noise = rng.normal(0.0, sigma, size=(n_draws, h))
    state_noise_cum = np.cumsum(state_noise, axis=1)

    # Total path per draw: AR mean + BSTS contribution + cumulative noise.
    total_paths = ar_path_mean[None, :] + bsts_paths + state_noise_cum

    # ---- 6. Conformal recalibration of the 90% band ---------------------
    # The raw posterior bands are typically too tight (BSTS posterior σ
    # underestimates because the model omits a lot of real-world drift).
    # Conformal α — learned per (target, model) on a validation slice —
    # scales the band away from the median until empirical coverage
    # matches the nominal 80% / 90% / 95%. α > 1 inflates, α < 1
    # shrinks.
    alpha = float(model.get("conformal_alpha", 1.0))
    median = np.median(total_paths, axis=0)
    q05 = np.quantile(total_paths, 0.05, axis=0)
    q95 = np.quantile(total_paths, 0.95, axis=0)
    q05_recal = median - alpha * (median - q05)
    q95_recal = median + alpha * (q95 - median)

    # ---- 7. Re-aggregate transform-space → level-space ------------------
    # For 'log_diff', this is last_level * cumprod(1 + step_returns).
    # For 'diff', this is last_level + cumsum(step_diffs).
    # For 'levels', this is the identity. The PM-facing output is in
    # level space (USD or bps), which is what they want to see.
    last_level = float(y_history.iloc[-1])
    level_median = np.asarray(
        transform.inverse_transform(median, last_level=last_level)
    ).astype(float)
    level_q05 = np.asarray(
        transform.inverse_transform(q05_recal, last_level=last_level)
    ).astype(float)
    level_q95 = np.asarray(
        transform.inverse_transform(q95_recal, last_level=last_level)
    ).astype(float)

    # ---- 7b. (Optional) ETF→OAS translation layer -----------------------
    # If the frozen pickle carries an `oas_overlay_translation` block, we
    # re-express the ETF-price forecast as an *OAS-bps* forecast using the
    # empirical regression slope from the FRED-overlap window
    # (see scripts/freeze_model_v4.py::_build_oas_overlay_translation).
    #
    # Algebra: implied ΔOAS_bps = slope · Δlog(ETF). Anchored on the latest
    # observed OAS level at freeze time. Slope is negative for both targets
    # (ETF price up → spread tighter), so the lo/hi bands flip: the lowest
    # OAS (tightest spread) corresponds to the highest ETF q95.
    #
    # This is a pure linear translation; no extra MCMC, no extra
    # uncertainty inflation. The proxy-quality caveat (Pearson) is
    # surfaced in `oas_overlay_meta` for honest reporting.
    oas_overlay = model.get("oas_overlay_translation")
    oas_outputs: dict[str, Any] = {}
    if oas_overlay is not None and last_level > 0:
        slope = float(oas_overlay["slope_bps_per_dlog"])
        last_oas = float(oas_overlay["last_oas_bps"])
        # Δlog from terminal observed level → forecast level, per step.
        # Guard against zero/negative levels in the band (extreme bands can
        # touch zero; clamp so log() stays defined).
        eps = 1e-9
        dlog_med = np.log(np.maximum(level_median, eps) / last_level)
        dlog_q05 = np.log(np.maximum(level_q05, eps) / last_level)
        dlog_q95 = np.log(np.maximum(level_q95, eps) / last_level)
        oas_path_median = last_oas + slope * dlog_med
        # Wire ETF-q95 → OAS-low and ETF-q05 → OAS-high (negative slope).
        # We then sort per-step to guarantee low ≤ high, which holds whenever
        # slope sign is consistent across the band (always true for these
        # targets, but we sort to be safe under future regimes).
        oas_path_a = last_oas + slope * dlog_q05
        oas_path_b = last_oas + slope * dlog_q95
        oas_path_lo = np.minimum(oas_path_a, oas_path_b)
        oas_path_hi = np.maximum(oas_path_a, oas_path_b)
        oas_outputs = {
            "oas_implied_median": float(oas_path_median[-1]),
            "oas_implied_band": (
                float(oas_path_lo[-1]), float(oas_path_hi[-1]),
            ),
            "oas_implied_path_median": oas_path_median.tolist(),
            "oas_implied_path_band_lo": oas_path_lo.tolist(),
            "oas_implied_path_band_hi": oas_path_hi.tolist(),
            "oas_overlay_meta": oas_overlay,
        }

    out_dict = {
        "target": model["target"],
        "target_transform": model["target_transform"],
        "as_of": as_of_ts.isoformat(),
        "horizon": h_label,
        "horizon_bd": h,
        "n_draws": int(n_draws),
        "conformal_alpha": alpha,
        # Scalars at the terminal step h — the headline forecast values.
        "median": float(median[-1]),
        "q05": float(q05_recal[-1]),
        "q95": float(q95_recal[-1]),
        "level_median": float(level_median[-1]),
        "level_band": (float(level_q05[-1]), float(level_q95[-1])),
        # Full paths (length h) for fan-chart plotting.
        "path_median": median.tolist(),
        "path_q05": q05_recal.tolist(),
        "path_q95": q95_recal.tolist(),
        "level_path_median": level_median.tolist(),
        "level_path_q05": level_q05.tolist(),
        "level_path_q95": level_q95.tolist(),
    }
    # Append OAS-implied fields if the frozen model carries the overlay block.
    out_dict.update(oas_outputs)
    return out_dict
