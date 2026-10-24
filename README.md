# propbot

A research harness for a rule-based intraday futures strategy aimed at a prop-firm
evaluation. It implements the pre-registration protocol: lock the rules, split the
data by year, grid-search only on development, gate, then validate and simulate the
firm's lifecycle. It does **not** place live orders.

## Quick start

```bash
pip install pandas numpy pytest
python -m pytest -q tests          # fill rules, look-ahead, lifecycle sanity
python -m propbot synth            # synthetic bars for testing code only (no edge by construction)
python -m propbot grid             # Step 3-4: dev-set grid + screening gate
python -m propbot validate         # Step 5: top candidate on year N-1, no tuning
python -m propbot holdout          # final proof on year N; run once
python -m propbot lifecycle --splits dev,validation   # Step 6: Monte Carlo eval + funded
```

Real data goes in `data/<SYMBOL>.csv` with columns `timestamp,open,high,low,close,volume`,
1-minute bars, timestamps in exchange time (CT for CME). You need at least 3 calendar years.

## Where things are

| File | What it does |
|---|---|
| `protocol.toml` | The pre-registered rules: hypothesis, session, stop/target, costs, grid, gate thresholds, firm rules. Its hash is stamped on every trial. |
| `propbot/engine.py` | Deterministic bar engine. Entries at next open, exits before entries, pessimistic same-bar stop/target, gaps fill at open, 1 tick slippage each side, daily-limit auto-liquidation, profit cap. |
| `propbot/strategies.py` | `orb` (opening-range breakout) and `vwap_revert` (VWAP fade, TWAP fallback with no volume). Signals at bar *i* use only bars before *i*. |
| `propbot/metrics.py` | Mean daily P&L, t-stat, Sharpe, max drawdown, win rate, share of profit from the best 1% of days. No-trade days count as zero. |
| `propbot/research.py` | Year splits, grid, gate (every market x timeframe must pass), locked validation/holdout with an access log. |
| `propbot/lifecycle.py` | Block-bootstrap Monte Carlo of evaluation then funded phase to N payouts, with trailing/static drawdown, daily limits, consistency and payout rules. |
| `results/` | `trials.csv` (every trial ever run), `grid_summary.csv`, `candidates.json`, `access_log.csv`. |

## Guard rails built in

* Validation and holdout refuse to run unless the params passed the dev gate and `protocol.toml`
  is unchanged since the grid ran. Every look is logged and repeat looks print a warning.
* The gate requires at least two markets and two timeframes, and every combination must have
  mean > 0, t above the threshold, and best-1%-day share < 50%.
* The lifecycle sim applies each phase's own daily limits to the raw trade stream, and checks
  each trade's worst open P&L against the daily limit and drawdown floor.

## Known simplifications

* One position at a time per instrument; fixed contracts.
* Lifecycle bootstrap treats days as exchangeable within 5-day blocks; it can't model regime change.
* Intraday trailing drawdown uses the best closed-trade equity during the day, not tick highs.
* Firm rules in `protocol.toml` are placeholders until you replace them with the firm's written rules (Step 8).
