import math

import numpy as np
import pandas as pd
import pytest

from propbot import data as D
from propbot import lifecycle as L
from propbot.engine import RunConfig, run
from propbot.instruments import get
from propbot.metrics import summarize
from propbot.strategies import orb, session_vwap, vwap_revert

ES = get("ES")   # $50/pt, 0.25 tick, $5 RT


def bars(rows, day="2024-03-04", start="08:30"):
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


def cfg(**kw):
    base = dict(instrument=ES, stop_usd=100.0, target_usd=100.0, slippage_ticks=1,
                last_entry="14:00", flatten="14:30")
    base.update(kw)
    return RunConfig(**base)


def sig_at(df, i, side=1):
    s = pd.Series(0, index=df.index)
    s.iloc[i] = side
    return s


def test_same_bar_stop_and_target_fills_at_worst_extreme():
    # $100 = 2 pts. Long entry 100 + 0.25 slip = 100.25; stop 98.25, target 102.25.
    df = bars([[100, 103, 97, 100, 10]])
    t = run(df, sig_at(df, 0), cfg()).trades.iloc[0]
    assert t.reason == "stop"
    assert t.exit == 97 - 0.25                       # bar low, minus slippage
    assert t.pnl == pytest.approx((96.75 - 100.25) * 50 - 5)


def test_same_bar_stop_price_option():
    df = bars([[100, 103, 97, 100, 10]])
    t = run(df, sig_at(df, 0), cfg(same_bar_fill="stop_price")).trades.iloc[0]
    assert t.exit == 98.25 - 0.25


def test_target_only_and_costs():
    df = bars([[100, 100.5, 99.5, 100, 10], [100, 103, 99.9, 102, 10]])
    t = run(df, sig_at(df, 0), cfg()).trades.iloc[0]
    assert t.reason == "target"
    assert t.pnl == pytest.approx((102.25 - 0.25 - 100.25) * 50 - 5)


def test_gap_through_stop_fills_at_open():
    df = bars([[100, 100.5, 99.5, 100, 10], [95, 96, 94, 95, 10]])
    t = run(df, sig_at(df, 0), cfg()).trades.iloc[0]
    assert t.reason == "stop_gap" and t.exit == 95 - 0.25


def test_short_side_mirror():
    df = bars([[100, 103, 97, 100, 10]])
    t = run(df, sig_at(df, 0, -1), cfg()).trades.iloc[0]
    assert t.side == -1 and t.exit == 103 + 0.25


def test_flatten_at_close():
    rows = [[100, 100.5, 99.5, 100, 10]] * 3
    df = bars(rows, start="14:28")
    t = run(df, sig_at(df, 0), cfg(last_entry="14:29")).trades.iloc[0]
    assert t.reason == "flatten" and t.exit_time == df.index[2] and t.exit == 100 - 0.25


def test_exit_before_entry_and_no_entry_on_exit_bar():
    df = bars([[100, 100.5, 99.5, 100, 10], [100, 100, 97, 98, 10], [98, 98.5, 97.5, 98, 10]])
    s = pd.Series([1, 1, 0], index=df.index)
    r = run(df, s, cfg(max_trades_per_day=5))
    assert len(r.trades) == 1      # bar 1 stopped us out; its signal must not re-enter on the same bar


def test_daily_loss_limit_liquidates():
    # Stop is $1000 away, daily limit $300: liquidation at the limit level comes first.
    df = bars([[100, 100.5, 99.5, 100, 10], [100, 100, 80, 85, 10]])
    r = run(df, sig_at(df, 0), cfg(stop_usd=1000, daily_loss_limit=300))
    t = r.trades.iloc[0]
    assert t.reason == "daily_limit"
    # level 94.35 -> first tick 94.25, minus 1 tick slippage = 94.00
    assert t.exit == pytest.approx(94.0)
    assert t.pnl == pytest.approx((94.0 - 100.25) * 50 - 5)


def test_profit_cap_halts_new_entries():
    df = bars([[100, 100.5, 99.5, 100, 10], [100, 103, 99.9, 102, 10], [102, 102, 102, 102, 10]])
    s = pd.Series([1, 0, 1], index=df.index)
    r = run(df, s, cfg(max_trades_per_day=5, daily_profit_cap=50))
    assert len(r.trades) == 1


def test_daily_frame_includes_no_trade_days():
    a = bars([[100, 100.5, 99.5, 100, 10]], day="2024-03-04")
    b = bars([[100, 100.5, 99.5, 100, 10]], day="2024-03-05")
    df = pd.concat([a, b])
    r = run(df, pd.Series(0, index=df.index), cfg())
    assert len(r.daily) == 2 and (r.daily["pnl"] == 0).all()


@pytest.mark.parametrize("fn,kw", [(orb, {"orb_minutes": 15}),
                                   (vwap_revert, {"lookback": 5, "threshold": 0.5, "window_minutes": 60})])
def test_no_lookahead(fn, kw):
    df = D.synthetic_bars("MES", start_year=2023, years=1, seed=3).iloc[: 390 * 3]
    base = fn(df, session_start="08:30", **kw)
    rng = np.random.default_rng(0)
    for cut in rng.integers(20, len(df) - 1, 15):
        future = df.copy()
        noise = rng.normal(0, 50, len(df) - cut)
        for col in ("open", "high", "low", "close"):
            future.iloc[cut:, future.columns.get_loc(col)] += noise
        future.iloc[cut:, future.columns.get_loc("volume")] = rng.integers(0, 1000, len(df) - cut)
        changed = fn(future, session_start="08:30", **kw)
        # bars up to and including `cut` are decided before bar `cut` closes
        assert (base.iloc[: cut + 1] == changed.iloc[: cut + 1]).all(), f"look-ahead at {cut}"


def test_vwap_falls_back_to_twap_without_volume():
    df = bars([[100, 102, 98, 100, 0], [100, 104, 100, 103, 0], [103, 104, 102, 103, 5]])
    v = session_vwap(df)
    assert v.iloc[1] == pytest.approx((100 + (104 + 100 + 103) / 3) / 2)
    assert v.iloc[2] == pytest.approx((104 + 102 + 103) / 3)


def test_top1pct_share():
    daily = pd.DataFrame({"pnl": [1000.0] + [-1.0] * 99})
    m = summarize(daily, pd.DataFrame(columns=["pnl"]))
    assert m["top1pct_share"] > 1   # the rest of the days lose money: a lottery ticket


def test_lifecycle_extremes():
    firm = {
        "evaluation": dict(account_size=50000, profit_target=3000, daily_loss_limit=1000,
                           daily_loss_is_breach=False, max_drawdown=2000, drawdown_type="eod_trailing",
                           trail_lock_at_start=True, daily_profit_cap=2000, consistency_max_day_share=0.5,
                           min_days=2, max_days=60),
        "funded": dict(account_size=50000, daily_loss_limit=500, daily_loss_is_breach=False,
                       max_drawdown=2000, drawdown_type="eod_trailing", trail_lock_at_start=True,
                       daily_profit_cap=2000, payout_min_days=5, payout_window=5, payout_green_days=4,
                       green_day_min=50, payout_min_profit=500, payout_max=2000, payouts_goal=2,
                       horizon_days=250),
    }
    winners = [[(300.0, -50.0)]] * 10
    out = L.simulate(winners, firm, paths=50, block=5, seed=1)
    assert out["eval_pass_rate"] == 1 and out["funded_goal_rate"] == 1
    losers = [[(-400.0, -450.0)]] * 10
    out = L.simulate(losers, firm, paths=50, block=5, seed=1)
    assert out["eval_pass_rate"] == 0


def test_drawdown_breach_uses_mae():
    rules = dict(daily_loss_limit=5000, daily_profit_cap=0, daily_loss_is_breach=False)
    out = L.play_day([(500.0, -2500.0)], rules, balance=50000, floor=48000)
    assert out.breached
