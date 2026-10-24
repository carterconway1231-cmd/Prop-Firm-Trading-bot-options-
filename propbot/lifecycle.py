"""Monte Carlo of the prop-firm lifecycle: evaluation, then funded account to N payouts.

Input is a list of trading days, each a list of (net_pnl, mae) trades in time order,
taken from a backtest run WITHOUT daily caps so each phase can apply its own limits.
Days are drawn with a block bootstrap. Intraday, a trade's MAE (worst open P&L) is
checked against the daily loss limit and the drawdown floor before its result counts,
so a trade that dipped through a limit is cut there even if it later recovered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DayOutcome:
    pnl: float
    breached: bool
    peak_intraday: float


def day_lists(trades: pd.DataFrame, all_days: pd.DatetimeIndex) -> list[list[tuple[float, float]]]:
    by_day: dict = {d: [] for d in all_days}
    for row in trades.sort_values("exit_time").itertuples():
        by_day.setdefault(row.day, []).append((float(row.pnl), float(row.mae)))
    return [by_day[d] for d in sorted(by_day)]


def play_day(trades, rules: dict, balance: float, floor: float) -> DayOutcome:
    dll = rules["daily_loss_limit"] or math.inf
    cap = rules.get("daily_profit_cap") or math.inf
    day, peak = 0.0, 0.0
    for pnl, mae in trades:
        if day <= -dll or day >= cap:
            break
        if balance + day + mae <= floor:
            return DayOutcome(floor - balance, True, peak)
        if day + mae <= -dll:
            day = -dll
            return DayOutcome(day, bool(rules.get("daily_loss_is_breach")), peak)
        day += pnl
        peak = max(peak, day)
    return DayOutcome(day, False, peak)


def _new_floor(rules: dict, start: float, peak_balance: float, floor: float) -> float:
    if rules["drawdown_type"] == "static":
        return floor
    f = max(floor, peak_balance - rules["max_drawdown"])
    if rules.get("trail_lock_at_start"):
        f = min(f, start)
    return max(f, floor)


def _sampler(n_days: int, block: int, rng: np.random.Generator):
    while True:
        s = int(rng.integers(n_days))
        for k in range(block):
            yield (s + k) % n_days


def run_evaluation(days, rules: dict, draw) -> tuple[bool, int]:
    start = rules["account_size"]
    bal, floor, peak = start, start - rules["max_drawdown"], start
    daily = []
    for n in range(1, rules["max_days"] + 1):
        out = play_day(days[next(draw)], rules, bal, floor)
        if rules["drawdown_type"] == "intraday_trailing":
            peak = max(peak, bal + out.peak_intraday)
        bal += out.pnl
        daily.append(out.pnl)
        if out.breached:
            return False, n
        peak = max(peak, bal)
        floor = _new_floor(rules, start, peak, floor)
        profit = bal - start
        share = rules.get("consistency_max_day_share") or 0
        consistent = share == 0 or (profit > 0 and max(daily) <= share * profit)
        if profit >= rules["profit_target"] and n >= rules["min_days"] and consistent:
            return True, n
    return False, rules["max_days"]


def run_funded(days, rules: dict, draw) -> dict:
    start = rules["account_size"]
    bal, floor, peak = start, start - rules["max_drawdown"], start
    recent: list[float] = []
    since, payouts, paid = 0, 0, 0.0
    for n in range(1, rules["horizon_days"] + 1):
        out = play_day(days[next(draw)], rules, bal, floor)
        if rules["drawdown_type"] == "intraday_trailing":
            peak = max(peak, bal + out.peak_intraday)
        bal += out.pnl
        if out.breached:
            return {"outcome": "breached", "payouts": payouts, "paid": paid, "days": n}
        peak = max(peak, bal)
        floor = _new_floor(rules, start, peak, floor)
        since += 1
        recent = (recent + [out.pnl])[-rules["payout_window"]:]
        green = sum(p >= rules["green_day_min"] for p in recent)
        profit = bal - start
        if (since >= rules["payout_min_days"] and green >= rules["payout_green_days"]
                and profit >= rules["payout_min_profit"]):
            amt = min(profit, rules["payout_max"])
            bal -= amt
            paid += amt
            payouts += 1
            since, recent = 0, []
            if payouts >= rules["payouts_goal"]:
                return {"outcome": "goal", "payouts": payouts, "paid": paid, "days": n}
    return {"outcome": "timeout", "payouts": payouts, "paid": paid, "days": rules["horizon_days"]}


def simulate(days, firm: dict, paths: int, block: int, seed: int) -> dict:
    if not days:
        raise ValueError("no days to sample")
    rng = np.random.default_rng(seed)
    ev, fu = firm["evaluation"], firm["funded"]
    passed, eval_days, funded = 0, [], []
    for _ in range(paths):
        draw = _sampler(len(days), block, rng)
        ok, n = run_evaluation(days, ev, draw)
        if ok:
            passed += 1
            eval_days.append(n)
            funded.append(run_funded(days, fu, draw))
    f = pd.DataFrame(funded) if funded else pd.DataFrame(columns=["outcome", "payouts", "paid", "days"])
    k = max(len(f), 1)
    return {
        "paths": paths,
        "eval_pass_rate": passed / paths,
        "eval_median_days": float(np.median(eval_days)) if eval_days else math.nan,
        "funded_goal_rate": float((f["outcome"] == "goal").sum() / k),
        "funded_breach_before_goal": float((f["outcome"] == "breached").sum() / k),
        "funded_timeout_rate": float((f["outcome"] == "timeout").sum() / k),
        "mean_paid_per_funded": float(f["paid"].mean()) if len(f) else 0.0,
        "overall_goal_rate": float((f["outcome"] == "goal").sum() / paths),
        "mean_daily_input": float(np.mean([sum(p for p, _ in d) for d in days])),
    }
