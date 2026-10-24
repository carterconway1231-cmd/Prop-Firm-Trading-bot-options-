"""Deterministic bar-by-bar backtest engine.

Fill rules (Step 3 of the protocol):
  * Entries fill at the bar's OPEN, plus slippage. Signals use only prior bars.
  * Exits for an open position are resolved before any new entry on that bar.
  * If the open gaps through the stop (or target), the fill is the open.
  * If stop and target are both inside one bar, the stop wins and fills at the
    bar's worst extreme (low for longs, high for shorts) by default.
  * Time flattens fill at the bar's close.
  * Every fill pays `slippage_ticks` against you; commission is per round trip.
  * Daily loss limit: the position is liquidated where the day's net P&L would hit
    -limit (like a firm's auto-liquidation), and the day is done.
  * Daily profit cap: once realized P&L reaches it, no new entries that day.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .instruments import Instrument


@dataclass
class RunConfig:
    instrument: Instrument
    stop_usd: float
    target_usd: float
    contracts: int = 1
    slippage_ticks: int = 1
    last_entry: str = "14:00"
    flatten: str = "14:30"
    max_trades_per_day: int = 1
    daily_loss_limit: float = math.inf
    daily_profit_cap: float = math.inf
    same_bar_fill: str = "bar_extreme"   # or "stop_price"


@dataclass
class Result:
    trades: pd.DataFrame
    daily: pd.DataFrame     # one row per session day in the data, including no-trade days


def _to_ticks(usd: float, inst: Instrument) -> float:
    ticks = max(1, round(usd / inst.tick_value))
    return ticks * inst.tick_size


def _hhmm(s: str) -> int:
    t = pd.Timestamp(s)
    return t.hour * 60 + t.minute


def run(df: pd.DataFrame, signals: pd.Series, cfg: RunConfig) -> Result:
    inst = cfg.instrument
    q = cfg.contracts
    pv = inst.point_value
    slip = cfg.slippage_ticks * inst.tick_size
    comm = inst.commission_rt * q
    stop_pts = _to_ticks(cfg.stop_usd, inst)
    tgt_pts = _to_ticks(cfg.target_usd, inst)
    last_entry, flatten = _hhmm(cfg.last_entry), _hhmm(cfg.flatten)

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sig = signals.reindex(df.index).fillna(0).to_numpy().astype(int)
    idx = df.index
    tmin = (idx.hour * 60 + idx.minute).to_numpy()
    days = idx.normalize()
    day_codes, day_labels = pd.factorize(days)

    trades = []
    pos = 0
    entry = stop = target = 0.0
    entry_i = 0
    mae = 0.0
    realized = 0.0
    n_today = 0
    halted = False
    cur_day = -1

    def net(exit_px: float) -> float:
        return (exit_px - entry) * pos * pv * q - comm

    def close_trade(i: int, exit_px: float, reason: str):
        nonlocal pos, realized, n_today, halted
        pnl = net(exit_px)
        trades.append((day_labels[day_codes[i]], idx[entry_i], idx[i], pos, entry, exit_px,
                       pnl, min(mae, pnl), reason))
        realized += pnl
        n_today += 1
        pos = 0
        if realized <= -cfg.daily_loss_limit + 1e-9 or realized >= cfg.daily_profit_cap - 1e-9:
            halted = True
        if reason == "daily_limit":
            halted = True

    def eff_stop() -> tuple[float, str]:
        """Tighter of the strategy stop and the daily-limit liquidation price."""
        if math.isinf(cfg.daily_loss_limit):
            return stop, "stop"
        liq = entry + pos * (-cfg.daily_loss_limit - realized + comm) / (pv * q)
        tick = inst.tick_size   # first tick at or beyond the limit level
        liq = (math.floor(liq / tick + 1e-9) if pos > 0 else math.ceil(liq / tick - 1e-9)) * tick
        if pos * (liq - stop) > 0:        # liquidation is closer to entry than the stop
            return liq, "daily_limit"
        return stop, "stop"

    def check_exit(i: int, use_open: bool) -> bool:
        """Resolve stop/target in bar i. Returns True if the position closed."""
        nonlocal mae
        s, s_reason = eff_stop()
        worst = l[i] if pos > 0 else h[i]
        best = h[i] if pos > 0 else l[i]
        if use_open and pos * (o[i] - s) <= 0:
            close_trade(i, o[i] - pos * slip, s_reason + "_gap")
            return True
        if use_open and pos * (o[i] - target) >= 0:
            close_trade(i, o[i] - pos * slip, "target_gap")
            return True
        stop_hit = pos * (worst - s) <= 0
        tgt_hit = pos * (best - target) >= 0
        if stop_hit:
            px = worst if (tgt_hit and cfg.same_bar_fill == "bar_extreme") else s
            close_trade(i, px - pos * slip, s_reason)
            return True
        mae = min(mae, net(worst - pos * slip))
        if tgt_hit:
            close_trade(i, target - pos * slip, "target")
            return True
        return False

    for i in range(len(o)):
        if day_codes[i] != cur_day:
            if pos:   # data ended before flatten time: exit at the previous bar's close
                close_trade(i - 1, c[i - 1] - pos * slip, "eod")
            cur_day = day_codes[i]
            realized, n_today, halted = 0.0, 0, False

        exited = False
        if pos:
            exited = check_exit(i, use_open=True)
            if pos and tmin[i] >= flatten:
                close_trade(i, c[i] - pos * slip, "flatten")
                exited = True

        if (not pos and not exited and not halted and sig[i] != 0
                and tmin[i] < last_entry and n_today < cfg.max_trades_per_day):
            pos = int(sig[i])
            entry = o[i] + pos * slip
            stop = entry - pos * stop_pts
            target = entry + pos * tgt_pts
            entry_i = i
            mae = -comm
            check_exit(i, use_open=False)

    if pos:
        close_trade(len(o) - 1, c[-1] - pos * slip, "eod")

    tdf = pd.DataFrame(trades, columns=["day", "entry_time", "exit_time", "side", "entry", "exit",
                                        "pnl", "mae", "reason"])
    daily = pd.DataFrame(index=pd.DatetimeIndex(day_labels, name="day"))
    if len(tdf):
        g = tdf.groupby("day")
        daily["pnl"] = g["pnl"].sum()
        daily["trades"] = g.size()
    else:
        daily["pnl"] = 0.0
        daily["trades"] = 0
    daily = daily.fillna(0.0)
    daily["trades"] = daily["trades"].astype(int)
    return Result(tdf, daily)
