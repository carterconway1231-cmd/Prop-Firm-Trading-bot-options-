"""Entry signals. Each strategy returns a Series of {-1, 0, +1} aligned to bars.

Contract: signal[i] is decided at the OPEN of bar i, so it may only use bars
0..i-1 (their closes are known by then). tests/test_no_lookahead.py checks this
by scrambling future bars and confirming earlier signals don't change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _minutes_since_open(idx: pd.DatetimeIndex, session_start: str) -> np.ndarray:
    s = pd.Timestamp(session_start)
    return (idx.hour * 60 + idx.minute - (s.hour * 60 + s.minute)).to_numpy()


def orb(df: pd.DataFrame, session_start: str, orb_minutes: int, **_) -> pd.Series:
    """Opening-range breakout: long when the prior bar closed above the range high,
    short when it closed below the range low. The range is every bar that opened in
    the first `orb_minutes`; signals start once all of those bars have closed."""
    day = df.index.normalize()
    mins = _minutes_since_open(df.index, session_start)
    in_range = mins < orb_minutes
    rh = df["high"].where(in_range).groupby(day).transform("max")
    rl = df["low"].where(in_range).groupby(day).transform("min")
    prev_close = df["close"].groupby(day).shift(1)
    live = (~in_range) & (mins >= orb_minutes)
    sig = np.where(live & (prev_close > rh), 1, np.where(live & (prev_close < rl), -1, 0))
    return pd.Series(sig, index=df.index, dtype=int)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative session VWAP through each bar's close. Uses typical price.
    Until the session has traded any volume, falls back to equal weights (TWAP)."""
    day = df.index.normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cv = df["volume"].groupby(day).cumsum()
    cpv = (tp * df["volume"]).groupby(day).cumsum()
    n = df.groupby(day).cumcount() + 1
    twap = tp.groupby(day).cumsum() / n
    return pd.Series(np.where(cv > 0, cpv / cv.replace(0, np.nan), twap), index=df.index)


def vwap_revert(df: pd.DataFrame, session_start: str, lookback: int = 10, threshold: float = 1.0,
                window_minutes: int = 30, **_) -> pd.Series:
    """Fade stretches from session VWAP during the first `window_minutes`.
    z = (close - vwap) / rolling std of (close - vwap), all through bar i-1."""
    day = df.index.normalize()
    dev = df["close"] - session_vwap(df)
    sd = dev.groupby(day).transform(lambda s: s.rolling(lookback, min_periods=lookback).std())
    z = (dev / sd.replace(0, np.nan)).groupby(day).shift(1)
    mins = _minutes_since_open(df.index, session_start)
    live = (mins < window_minutes) & z.notna().to_numpy()
    sig = np.where(live & (z > threshold), -1, np.where(live & (z < -threshold), 1, 0))
    return pd.Series(sig, index=df.index, dtype=int)


STRATEGIES = {"orb": orb, "vwap_revert": vwap_revert}

# Grid keys that are strategy parameters (the rest are risk parameters).
STRATEGY_PARAMS = {"orb": {"orb_minutes"}, "vwap_revert": {"lookback", "threshold", "window_minutes"}}
