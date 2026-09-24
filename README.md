# Quant Factor Backtester

A walk-forward equity factor backtester with an honesty-first design: point-in-time signals, transaction costs on drifted turnover, and dollar-neutral portfolio construction. CLI + zero-dependency web UI with charts.

> **SIMULATED DATA** — every price in `data/` is synthetically generated. Not real market data. Not investment advice.

## Quickstart

```bash
python demo.py                        # one-command demo: all factors vs benchmark
python cli.py run --factor momentum   # single factor, full metrics
python cli.py run --factor reversal --long-only --cost-bps 25 --ic
python cli.py compare                 # all factors + equal-weight benchmark
python server.py                      # charts at http://localhost:8000
python -m unittest discover -s tests  # 28 hermetic tests
```

No third-party dependencies — Python 3.10+ standard library only.

## What it does

Three classic cross-sectional equity factors, each a standard academic definition:

| Factor | Definition |
|---|---|
| `momentum` | 12-month return skipping the most recent month (12-1) |
| `reversal` | Prior-month return, negated (short-term reversal) |
| `lowvol` | Trailing 63-day volatility, negated (low-volatility anomaly) |

At each month-end rebalance the engine ranks all names on the factor, goes long the top quintile and short the bottom quintile (equal-weighted, dollar-neutral: 0.5x equity per leg), and holds until the next rebalance. A `--long-only` mode is also available.

## The methodology (the honest version)

Most notebook backtests lie to you in three ways. This one doesn't:

1. **No lookahead, by construction.** `factors.compute_signals()` truncates every price series at the rebalance date *before* any factor math runs. Future data cannot leak in even by accident — and `TestNoLookahead` proves it: corrupting all future prices 1000x leaves every signal bit-identical.
2. **Walk-forward, always.** Every return is out-of-sample. Signals for window T use only data ≤ T's rebalance date. There is no "fit on everything, report on everything" step.
3. **Costs on drifted turnover.** Previous weights are evolved by realized returns before turnover is measured, so the 10 bps/trade cost reflects what actually traded — including day-one entry costs. Rebalance-day costs are folded into that day's return, so the reported return series compounds exactly to the equity curve. (An early version sized positions off pre-cost equity, which silently levered the book ~3% whenever costs were nonzero. The test suite caught it — see below.)

Also reported: Sharpe, Sortino, max drawdown, win rate, per-rebalance turnover, total cost drag, and the **information coefficient** (Spearman rank correlation between signal and next-month return — whether the factor predicts anything, independent of portfolio construction).

## What the tests caught (dev-loop story)

The 28-test suite found four real bugs during this build:

- **Dict-iteration bug (×2):** `sorted(series)` on a `{date: close}` dict yields dates, not pairs — `TypeError` in the return lookup, in both `run()` and `run_equal_weight()`.
- **Rank-tie bug:** the Spearman IC gave a perfect 1.0 for a constant signal because ties got distinct ranks by index. Fixed with proper average-tie ranking; constant signals now correctly score 0.0.
- **Phantom leverage:** legs were sized off pre-cost equity, so any nonzero cost setting silently levered the portfolio (costly runs showed *higher* gross returns — the test asserting costs reduce returns caught it). Legs are now sized off post-cost equity.
- **Equity/returns drift:** rebalance costs were deducted from equity but never recorded as returns, so the return series didn't compound to the equity curve. Costs are now folded into the rebalance day's return.

## Honest results

On the bundled synthetic universe (regime-switching GBM, ~4 years, 12 names), the factors roughly break even while the equal-weight benchmark wins — because the generator's regime half-life (~6 months) is shorter than momentum's 12-month lookback, so the signal is usually stale. That's the point: the harness tells you the truth instead of flattering the factor. Swap in your own CSVs (same format) to test a real universe.

## Structure

```
data_gen.py    deterministic synthetic OHLC generator + CSV loader
factors.py     momentum / reversal / lowvol (point-in-time by construction)
backtest.py    walk-forward engine: monthly rebalance, costs, dollar-neutral L/S
metrics.py     Sharpe, Sortino, drawdown, IC, win rate
cli.py         generate / run / compare commands
server.py      zero-dependency web UI (equity curves, drawdowns, rebalance log)
demo.py        one-command demo
data/          12 synthetic symbols, ~4y daily (SIMULATED DATA banners in-file)
tests/         28 hermetic tests (deterministic, no network)
```

## License

MIT — see [LICENSE](LICENSE).
