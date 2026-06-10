"""Target-side transforms (``levels`` / ``diff`` / ``log_diff``) with inverses.

v1 / v2 fit BSTS in *level* space because BSTS's structural components
(local linear trend, seasonal) naturally accommodate non-stationary y. v3
moves to *change* space for the OAS targets — PMs care about "Δ HY OAS next
week = +12 bps," not "level next week = 387 bps."

v2.1 wired ``log_diff`` for the ETF targets ad-hoc in the data-loading
layer. v3 promotes this to a first-class preprocessing step so OAS (bps)
and ETF (USD) round-trip consistently and the inverse transform is
available at inference time for re-aggregating back to level space.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TransformKind = Literal["levels", "diff", "log_diff"]


class TargetTransform:
    """Applies and inverts target transforms on a univariate level series.

    ``fit`` learns no parameters; the transform is deterministic. ``last_level``
    is captured at fit time so a single ``inverse_transform`` call (on
    horizon-h forecasts) reproduces the cumulative path from a known anchor.

    Parameters
    ----------
    kind : {"levels", "diff", "log_diff"}
        ``"levels"`` is a no-op (used for parity / backward compatibility).
        ``"diff"`` returns first-differences ``y_t - y_{t-1}``.
        ``"log_diff"`` returns log-returns ``log(y_t) - log(y_{t-1})``.
    """

    def __init__(self, kind: TransformKind = "diff"):
        if kind not in ("levels", "diff", "log_diff"):
            raise ValueError(f"kind={kind!r} not in ('levels', 'diff', 'log_diff')")
        self.kind: TransformKind = kind
        self.last_level_: float | None = None
        self.last_index_: pd.Timestamp | None = None

    def fit(self, level_series: pd.Series) -> TargetTransform:
        """Remember the final level + its timestamp so we can invert later."""
        if level_series.empty:
            raise ValueError("TargetTransform.fit got an empty series")
        self.last_level_ = float(level_series.iloc[-1])
        self.last_index_ = level_series.index[-1]
        return self

    def transform(self, level_series: pd.Series) -> pd.Series:
        """Apply the transform; the first observation becomes NaN for diff/log_diff."""
        if self.kind == "levels":
            return level_series.copy()
        if self.kind == "diff":
            return level_series.diff()
        if self.kind == "log_diff":
            if (level_series <= 0).any():
                raise ValueError(
                    "log_diff requires strictly positive levels; "
                    f"min={level_series.min()}"
                )
            return np.log(level_series).diff()
        # Should be unreachable thanks to __init__ validation.
        raise ValueError(self.kind)

    def fit_transform(self, level_series: pd.Series) -> pd.Series:
        return self.fit(level_series).transform(level_series)

    def inverse_transform(
        self,
        transformed: pd.Series | pd.DataFrame | float | np.ndarray,
        last_level: float | None = None,
    ) -> pd.Series | pd.DataFrame | float | np.ndarray:
        """Re-aggregate a transformed forecast back to level space.

        Parameters
        ----------
        transformed :
            Either a single Δ (or Δlog) value, a 1-D path of such values, a
            ``pd.Series`` keyed by future timestamps, or a ``pd.DataFrame``
            whose columns are scenario paths (e.g. posterior draws).
        last_level : float, optional
            The level anchor (``y_T``) to cumulate from. Defaults to the value
            captured in ``fit``; raises if neither is set.

        Returns
        -------
        Same shape as ``transformed`` but in level units.
        """
        anchor = self.last_level_ if last_level is None else float(last_level)
        if anchor is None:
            raise RuntimeError(
                "TargetTransform.inverse_transform needs a level anchor — "
                "call fit() first or pass last_level=..."
            )

        if self.kind == "levels":
            return transformed

        # Scalar fast path.
        if np.isscalar(transformed):
            if self.kind == "diff":
                return anchor + float(transformed)
            return anchor * float(np.exp(transformed))

        # Array / Series / DataFrame path.
        arr = np.asarray(transformed)
        if self.kind == "diff":
            cumulative = np.cumsum(arr, axis=0)
            out_arr = anchor + cumulative
        else:  # log_diff
            cumulative = np.cumsum(arr, axis=0)
            out_arr = anchor * np.exp(cumulative)

        if isinstance(transformed, pd.DataFrame):
            return pd.DataFrame(out_arr, index=transformed.index,
                                columns=transformed.columns)
        if isinstance(transformed, pd.Series):
            return pd.Series(out_arr, index=transformed.index, name=transformed.name)
        return out_arr

    def __repr__(self) -> str:
        return (
            f"TargetTransform(kind={self.kind!r}, "
            f"last_level={self.last_level_}, last_index={self.last_index_})"
        )
