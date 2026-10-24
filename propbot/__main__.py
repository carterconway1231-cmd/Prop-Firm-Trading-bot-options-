"""Command line: python -m propbot <command>

  synth        write synthetic 1-minute bars to data/ (for testing code only)
  grid         run the pre-registered grid on the development set + screening gate
  validate     run the top dev candidate on the validation year, no tuning
  holdout      same, on the most recent year (final proof; run once)
  lifecycle    Monte Carlo the evaluation + funded phases for a candidate
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import data as D
from . import lifecycle as L
from . import research as R


def cmd_synth(args):
    R.DATA.mkdir(exist_ok=True)
    proto = R.load_protocol()
    for i, sym in enumerate(proto["data"]["instruments"]):
        df = D.synthetic_bars(sym, start_year=args.start_year, years=args.years, seed=args.seed + i)
        df.to_csv(R.DATA / f"{sym}.csv")
        print(f"wrote data/{sym}.csv  {len(df):,} bars  {df.index[0]} .. {df.index[-1]}")


def _print_rows(rows):
    cols = ["instrument", "timeframe", "days", "trades", "win_rate", "mean_daily", "t_stat", "sharpe",
            "max_dd", "top1pct_share"]
    print(pd.DataFrame(rows)[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


def cmd_grid(args):
    proto = R.load_protocol()
    out = R.grid_search(proto)
    n = len(out)
    print(f"protocol {proto['_hash']}  strategy={proto['hypothesis']['name']}  {n} parameter sets "
          f"x {len(proto['data']['instruments'])} markets x {len(proto['data']['timeframes'])} timeframes")
    print(out.head(10).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    k = int(out["passed"].sum())
    print(f"\n{k}/{n} passed the development gate. Every trial is in results/trials.csv.")
    if k == 0:
        print("Nothing passed. Per Step 9, accept it and move to a different hypothesis.")


def cmd_locked(args, split):
    proto = R.load_protocol()
    params = json.loads(args.params) if args.params else None
    ok, why, rows, prior = R.locked_run(proto, split, params)
    if prior:
        print(f"WARNING: {split} has been run {prior} time(s) before. Repeated looks turn it into "
              f"a development set.", file=sys.stderr)
    print(f"{split}  params={rows[0]['params']}")
    _print_rows(rows)
    print("\nPASS" if ok else "\nFAIL: " + "; ".join(why))


def cmd_lifecycle(args):
    proto = R.load_protocol()
    params = json.loads(args.params) if args.params else json.loads(
        (R.RESULTS / "candidates.json").read_text())["candidates"][0]
    tf = args.timeframe or proto["data"]["timeframes"][0]
    merged: dict = {}
    for split in args.splits.split(","):
        for sym in proto["data"]["instruments"]:
            res = R.run_one(proto, params, sym, tf, split, daily_limits=False)
            for d, day in zip(sorted(set(res.daily.index)), L.day_lists(res.trades, res.daily.index)):
                merged.setdefault(d, []).extend(day)
    days = [merged[d] for d in sorted(merged)]
    lc = proto["lifecycle"]
    out = L.simulate(days, proto["firm"], args.paths or lc["paths"], lc["block_days"], lc["seed"])
    print(f"lifecycle  params={json.dumps(params, sort_keys=True)}  markets={proto['data']['instruments']} "
          f"tf={tf}m  splits={args.splits}  days={len(days)}")
    for k, v in out.items():
        print(f"  {k:28s} {v:,.3f}" if isinstance(v, float) else f"  {k:28s} {v}")
    good = out["eval_pass_rate"] > 0.70 and out["funded_breach_before_goal"] < 0.30
    print("\nMeets Step 6 targets (pass > 70%, funded breach < 30%)" if good
          else "\nDoes NOT meet Step 6 targets (pass > 70%, funded breach < 30%)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="propbot")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--start-year", type=int, default=2022)
    s.add_argument("--years", type=int, default=4)
    s.add_argument("--seed", type=int, default=1)
    sub.add_parser("grid")
    for name in ("validate", "holdout"):
        v = sub.add_parser(name)
        v.add_argument("--params", help="JSON params; must be a dev-gate candidate")
    lc = sub.add_parser("lifecycle")
    lc.add_argument("--params", help="JSON params (default: top dev candidate)")
    lc.add_argument("--splits", default="dev", help="comma list of dev,validation,holdout")
    lc.add_argument("--timeframe", type=int)
    lc.add_argument("--paths", type=int)
    a = p.parse_args(argv)
    if a.cmd == "synth":
        cmd_synth(a)
    elif a.cmd == "grid":
        cmd_grid(a)
    elif a.cmd in ("validate", "holdout"):
        cmd_locked(a, "validation" if a.cmd == "validate" else "holdout")
    else:
        cmd_lifecycle(a)


if __name__ == "__main__":
    main()
