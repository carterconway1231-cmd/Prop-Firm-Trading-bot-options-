"""Grid search on the development set, the screening gate, and locked validation/holdout runs."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data as D
from .engine import RunConfig, run
from .instruments import get as get_instrument
from .metrics import summarize
from .strategies import STRATEGIES, STRATEGY_PARAMS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def load_protocol(path: Path = ROOT / "protocol.toml") -> dict:
    raw = path.read_bytes()
    proto = tomllib.loads(raw.decode())
    proto["_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    return proto


def grid_points(proto: dict) -> list[dict]:
    g = proto["grid"]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


_bars_cache: dict = {}


def bars(proto: dict, symbol: str, tf: int, split: str) -> pd.DataFrame:
    key = (symbol, tf, split, proto["session"]["start"], proto["session"]["end"])
    if key not in _bars_cache:
        df = D.load_bars(DATA / f"{symbol}.csv")
        df = D.session_filter(df, proto["session"]["start"], proto["session"]["end"])
        parts = D.split_years(df)
        for name in ("dev", "validation", "holdout"):
            _bars_cache[(symbol, tf, name, *key[3:])] = D.resample(parts[name], tf)
    return _bars_cache[key]


def run_one(proto: dict, params: dict, symbol: str, tf: int, split: str, daily_limits: bool = True):
    strat = proto["hypothesis"]["name"]
    s, r = proto["session"], proto["risk"]
    df = bars(proto, symbol, tf, split)
    sp = {k: v for k, v in params.items() if k in STRATEGY_PARAMS[strat]}
    sig = STRATEGIES[strat](df, session_start=s["start"], **sp)
    cfg = RunConfig(
        instrument=get_instrument(symbol),
        stop_usd=params.get("stop_usd", r["stop_usd"]),
        target_usd=params.get("target_usd", r["target_usd"]),
        contracts=r["contracts"],
        slippage_ticks=r["slippage_ticks"],
        last_entry=s["last_entry"],
        flatten=s["flatten"],
        max_trades_per_day=s["max_trades_per_day"],
        daily_loss_limit=r["daily_loss_limit"] if daily_limits else math.inf,
        daily_profit_cap=r["daily_profit_cap"] if daily_limits else math.inf,
    )
    return run(df, sig, cfg)


TRIAL_FIELDS = ["logged_at", "protocol", "split", "strategy", "instrument", "timeframe", "params",
                "days", "trades", "win_rate", "mean_daily", "t_stat", "sharpe", "max_dd", "total",
                "top1pct_share"]


def log_trial(row: dict):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "trials.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in TRIAL_FIELDS})


def evaluate(proto: dict, params: dict, split: str) -> list[dict]:
    rows = []
    for sym in proto["data"]["instruments"]:
        for tf in proto["data"]["timeframes"]:
            res = run_one(proto, params, sym, tf, split)
            m = summarize(res.daily, res.trades)
            row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "protocol": proto["_hash"], "split": split, "strategy": proto["hypothesis"]["name"],
                   "instrument": sym, "timeframe": tf, "params": json.dumps(params, sort_keys=True), **m}
            log_trial(row)
            rows.append(row)
    return rows


def gate(rows: list[dict], min_t: float, max_share: float, all_positive: bool) -> tuple[bool, list[str]]:
    """Every instrument x timeframe combo must clear the bar, not just the average."""
    why = []
    combos = {(r["instrument"], r["timeframe"]) for r in rows}
    if len({c[0] for c in combos}) < 2 or len({c[1] for c in combos}) < 2:
        why.append("needs at least two markets and two timeframes")
    for r in rows:
        tag = f"{r['instrument']}/{r['timeframe']}m"
        if all_positive and r["mean_daily"] <= 0:
            why.append(f"{tag} mean {r['mean_daily']:.2f} <= 0")
        if r["t_stat"] < min_t:
            why.append(f"{tag} t {r['t_stat']:.2f} < {min_t}")
        if not r["top1pct_share"] < max_share:
            why.append(f"{tag} top-1% share {r['top1pct_share']:.2f} >= {max_share}")
    return (not why), why


def grid_search(proto: dict) -> pd.DataFrame:
    g = proto["gate"]
    summary = []
    for params in grid_points(proto):
        rows = evaluate(proto, params, "dev")
        ok, why = gate(rows, g["min_t_dev"], g["max_top1pct_share"], g["require_all_combos_positive"])
        summary.append({"params": json.dumps(params, sort_keys=True), "passed": ok,
                        "min_t": min(r["t_stat"] for r in rows),
                        "mean_daily_avg": sum(r["mean_daily"] for r in rows) / len(rows),
                        "max_top_share": max(r["top1pct_share"] for r in rows),
                        "reasons": "; ".join(why[:4])})
    out = pd.DataFrame(summary).sort_values("min_t", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "grid_summary.csv", index=False)
    passed = [json.loads(p) for p in out.loc[out["passed"], "params"]]
    (RESULTS / "candidates.json").write_text(json.dumps(
        {"protocol": proto["_hash"], "candidates": passed}, indent=2))
    return out


def _access_log(split: str, proto: dict, params: dict) -> int:
    path = RESULTS / "access_log.csv"
    prior = 0
    if path.exists():
        prior = sum(1 for line in path.read_text().splitlines()[1:] if f",{split}," in line)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["logged_at", "split", "protocol", "params"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), split, proto["_hash"],
                    json.dumps(params, sort_keys=True)])
    return prior


def locked_run(proto: dict, split: str, params: dict | None = None) -> tuple[bool, list[str], list[dict], int]:
    """Validation or holdout with a candidate that passed the dev gate. No tuning."""
    cands = json.loads((RESULTS / "candidates.json").read_text())
    if params is None:
        if not cands["candidates"]:
            raise SystemExit("No candidate passed the development gate. Per Step 9: don't touch "
                             "validation; move to a different hypothesis.")
        params = cands["candidates"][0]
    elif params not in cands["candidates"]:
        raise SystemExit("Those params did not pass the dev gate; refusing to peek at locked data.")
    if cands["protocol"] != proto["_hash"]:
        raise SystemExit("protocol.toml changed since the grid ran. Re-run the grid first.")
    prior = _access_log(split, proto, params)
    rows = evaluate(proto, params, split)
    g = proto["gate"]
    ok, why = gate(rows, g["min_t_validation"], g["max_top1pct_share"], True)
    return ok, why, rows, prior
