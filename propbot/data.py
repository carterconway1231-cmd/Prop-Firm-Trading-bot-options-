"""Bar loading, resampling, year splits, and a synthetic generator for code testing.

Expected CSV columns: timestamp,open,high,low,close,volume
Timestamps must already be in exchange time (CT for CME), one row per bar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import get as get_instrument


def load_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} bars have high/low inconsistent with open/close")
    return df


def session_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep bars whose open time is in [start, end)."""
    t = df.index.time
    s, e = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    return df[(t >= s) & (t < e)]


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


def split_years(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """dev = years 1..N-2, validation = N-1, holdout = N."""
    years = sorted(df.index.year.unique())
    if len(years) < 3:
        raise ValueError(f"Need at least 3 calendar years of data, have {years}")
    y = df.index.year
    return {
        "dev": df[y <= years[-3]],
        "validation": df[y == years[-2]],
        "holdout": df[y == years[-1]],
        "_years": years,
    }


def synthetic_bars(symbol: str, start_year: int = 2021, years: int = 4, seed: int = 0,
                   start_price: float | None = None) -> pd.DataFrame:
    """Random-walk 1-minute RTH bars with intraday volatility/volume smile.

    There is no edge in this data by construction. Use it to test code, never to
    draw conclusions. A strategy that 'passes' on it is a bug or an overfit.
    """
    inst = get_instrument(symbol)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{start_year}-01-01", f"{start_year + years - 1}-12-31")
    minutes = pd.timedelta_range("08:30:00", "14:59:00", freq="1min")
    n = len(minutes)
    smile = 1.0 + 1.5 * np.exp(-np.arange(n) / 30.0) + 0.5 * np.exp(-(n - np.arange(n)) / 20.0)
    price = start_price or {"MES": 4000.0, "ES": 4000.0, "MNQ": 13000.0, "NQ": 13000.0}.get(symbol.upper(), 2000.0)
    per_min_vol = price * 0.00045
    frames = []
    for d in days:
        price *= np.exp(rng.normal(0, 0.004))   # overnight gap
        rets = rng.standard_t(4, n) * per_min_vol * smile / np.sqrt(2)
        close = price + np.cumsum(rets)
        open_ = np.r_[price, close[:-1]]
        wick = np.abs(rng.normal(0, per_min_vol * 0.6, (2, n))) * smile
        high = np.maximum(open_, close) + wick[0]
        low = np.minimum(open_, close) - wick[1]
        tick = inst.tick_size
        o, h, l, c = (np.round(x / tick) * tick for x in (open_, high, low, close))
        h = np.maximum.reduce([h, o, c])
        l = np.minimum.reduce([l, o, c])
        vol = rng.poisson(200 * smile).astype(float)
        vol[rng.random(n) < 0.01] = 0.0          # occasional empty bars exercise the TWAP fallback
        frames.append(pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol},
                                   index=d + minutes))
        price = c[-1]
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df
