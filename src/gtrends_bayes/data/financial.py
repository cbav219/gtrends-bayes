"""Financial data fetcher (FRED for OAS / VIX / UST yields, yfinance for cross-checks)."""

from __future__ import annotations

import os
from datetime import date
from typing import Literal

import pandas as pd

from gtrends_bayes.logging import get_logger

Source = Literal["fred", "yfinance", "derived"]

log = get_logger(__name__)


def _fred_client():
    """Construct a fredapi.Fred client from the FRED_API_KEY env var."""
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return Fred(api_key=api_key)


def fetch_target(
    name: str,
    source: Source,
    ticker: str,
    start: date,
    end: date,
) -> pd.Series:
    """Fetch a single financial target series from FRED or yfinance.

    Parameters
    ----------
    name : str
        Logical name used downstream (e.g. ``"HY_OAS"``).
    source : {"fred", "yfinance", "derived"}
        ``"derived"`` is reserved for synthetic series (e.g. yield-curve
        slope) computed by the feature layer; this function refuses it.
    ticker : str
    start, end : date

    Returns
    -------
    pandas.Series
        Daily series, named ``name``, indexed by ``DatetimeIndex``.
    """
    if source == "fred":
        client = _fred_client()
        log.info("fetching FRED series %s (%s..%s)", ticker, start, end)
        s = client.get_series(ticker, observation_start=start, observation_end=end)
        s = s.rename(name)
        s.index = pd.to_datetime(s.index)
        return s.dropna()

    if source == "yfinance":
        import yfinance as yf

        log.info("fetching yfinance ticker %s (%s..%s)", ticker, start, end)
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"yfinance returned empty history for {ticker}")
        s = hist["Close"].rename(name)
        # yfinance returns tz-aware; strip for consistency with FRED.
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s

    raise ValueError(
        f"source={source!r} not supported by fetch_target; 'derived' must be"
        " computed in the features layer from previously-fetched series."
    )


def resample_weekly(series: pd.Series, anchor: str = "SUN") -> pd.Series:
    """Resample a daily series to weekly bars aligned on ``anchor`` (default Sunday).

    Spreads (FRED) and equities (yfinance) trade Mon–Fri. Resampling to
    ``W-SUN`` with ``.last()`` bins each Mon–Sun week into the following
    Sunday timestamp, which matches the timestamp Google uses for its
    weekly-bar Trends responses.
    """
    rule = f"W-{anchor.upper()}"
    return series.resample(rule).last().dropna()
