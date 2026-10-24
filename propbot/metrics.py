"""Daily-P&L statistics used by the screening gate."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    x = daily["pnl"].to_numpy(dtype=float)
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else math.nan
    t = mean / se if se and se > 0 else 0.0
    sharpe = mean / sd * math.sqrt(252) if sd > 0 else 0.0
    equity = np.cumsum(x)
    max_dd = float((np.maximum.accumulate(np.r_[0.0, equity]) - np.r_[0.0, equity]).max()) if n else 0.0
    total = float(x.sum())
    k = max(1, math.ceil(0.01 * n))
    top = float(np.sort(x)[-k:].sum()) if n else 0.0
    # Share of total profit from the best 1% of days. >1 (or inf) means the rest lose money.
    top_share = top / total if total > 0 else math.inf
    traded = trades["pnl"].to_numpy(dtype=float) if len(trades) else np.array([])
    return {
        "days": n,
        "trades": int(len(traded)),
        "win_rate": float((traded > 0).mean()) if len(traded) else 0.0,
        "mean_daily": mean,
        "t_stat": t,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total": total,
        "top1pct_share": top_share,
    }
