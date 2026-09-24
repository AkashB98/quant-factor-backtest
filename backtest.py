"""Walk-forward factor backtest engine.

Methodology (the honest version):
- WALK-FORWARD: the sample is split into monthly rebalance windows. Signals
  for window T are computed from data <= the rebalance date only
  (see factors.compute_signals). There is no "train on everything, test on
  everything" step — every return is out-of-sample by construction.
- NO LOOKAHEAD: prices are truncated at the rebalance date before factor
  math runs. A regression test proves shuffling future data changes nothing.
- COSTS: trades pay `cost_bps` per unit of turnover. Previous weights are
  drifted by realized returns before turnover is measured, so costs reflect
  what actually traded — including the cost of entering positions on day 1.
- DOLLAR-NEUTRAL long/short: 0.5x equity long leg, 0.5x equity short leg
  (1.0x gross). Long-only mode also available.

Nothing here is investment advice. All data is synthetic.
"""

import statistics
from dataclasses import dataclass, field
from datetime import date

from factors import compute_signals
from metrics import summary as metrics_summary


@dataclass
class Config:
    factor: str = "momentum"      # momentum | reversal | lowvol
    top_frac: float = 0.2         # long the top quintile
    bottom_frac: float = 0.2      # short the bottom quintile
    cost_bps: float = 10.0        # transaction cost per unit turnover
    long_short: bool = True       # False -> long-only top quintile
    min_history: int = 260        # trading days before first rebalance


@dataclass
class BacktestResult:
    factor: str
    config: Config
    dates: list[date]                       # trading days in sample
    daily_returns: list[float]              # net of costs, per trading day
    equity: list[float]
    rebalances: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    avg_turnover: float = 0.0
    total_cost_drag: float = 0.0            # sum of costs as fraction of equity


def month_end_dates(dates: list[date]) -> list[date]:
    """Last trading day of each calendar month present in `dates`."""
    ends: list[date] = []
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        if nxt is None or nxt.month != d.month or nxt.year != d.year:
            ends.append(d)
    return ends


def _zscore(values: dict[str, float]) -> dict[str, float]:
    xs = list(values.values())
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(set(xs)) > 1 else 1.0
    return {k: (v - mu) / sd for k, v in values.items()}


def target_weights(signals: dict[str, dict[str, float]], factor: str,
                   cfg: Config) -> tuple[dict[str, float], dict[str, float]]:
    """Quintile portfolios from cross-sectional z-scores.

    Returns (long_w, short_w): each maps symbol -> weight WITHIN its leg
    (weights sum to 1 within a leg). Empty dicts when too few names.
    """
    vals = {s: f[factor] for s, f in signals.items() if factor in f}
    if len(vals) < 5:
        return {}, {}
    z = _zscore(vals)
    ranked = sorted(z, key=lambda s: z[s])
    n = len(ranked)
    n_long = max(1, int(n * cfg.top_frac))
    n_short = max(1, int(n * cfg.bottom_frac)) if cfg.long_short else 0
    longs = ranked[-n_long:]
    shorts = ranked[:n_short]
    long_w = {s: 1.0 / len(longs) for s in longs}
    short_w = {s: 1.0 / len(shorts) for s in shorts}
    return long_w, short_w


def _drift_leg(prev_w: dict[str, float],
               rets: dict[str, float]) -> dict[str, float]:
    """Evolve within-leg weights by realized returns (renormalized)."""
    grown = {s: w * (1.0 + rets.get(s, 0.0)) for s, w in prev_w.items()}
    tot = sum(grown.values())
    if tot <= 0:
        return {}
    return {s: w / tot for s, w in grown.items()}


def _turnover(old: dict[str, float], new: dict[str, float]) -> float:
    syms = set(old) | set(new)
    return sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in syms)


def run(price_panel: dict[str, list[tuple[date, float]]],
        cfg: Config) -> BacktestResult:
    """Walk-forward backtest. Returns net-of-cost daily returns + diagnostics."""
    # Common trading calendar: intersection of all symbols' dates.
    all_dates = sorted({d for series in price_panel.values() for d, _ in series})
    closes: dict[str, dict[date, float]] = {
        s: dict(series) for s, series in price_panel.items()
    }
    dates = [d for d in all_dates
             if all(d in closes[s] for s in price_panel)]
    if len(dates) < cfg.min_history + 60:
        raise ValueError("Not enough history for a walk-forward backtest")

    reb_dates = [d for d in month_end_dates(dates)
                 if dates.index(d) >= cfg.min_history]
    if not reb_dates:
        raise ValueError("No rebalance dates with enough history")

    equity = 1.0
    equity_curve = [equity]
    daily_returns: list[float] = []
    out_dates = [dates[0]]

    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    long_leg = 0.0    # dollar notional of each leg; re-set at rebalances
    short_leg = 0.0
    rebalances: list[dict] = []
    total_cost = 0.0
    turnovers: list[float] = []

    # Fast return lookup: {symbol: {date: next-day simple return}}
    rets: dict[str, dict[date, float]] = {}
    for s, series in closes.items():
        sd = sorted(series.items())
        rets[s] = {sd[i][0]: sd[i + 1][1] / sd[i][1] - 1.0
                   for i in range(len(sd) - 1)}

    for ri, rdate in enumerate(reb_dates):
        ridx = dates.index(rdate)

        # 1. Point-in-time signals (never sees the future).
        signals = compute_signals(price_panel, rdate, cfg.min_history)
        long_w, short_w = target_weights(signals, cfg.factor, cfg)

        # 2. Drift previous weights to today, measure turnover, pay costs.
        if ri > 0:
            # Evolve last rebalance's weights by realized returns so the
            # turnover we charge reflects what actually traded.
            d_long = prev_long_w
            d_short = prev_short_w
            for j in range(prev_ridx, ridx):
                jr = {s: rets[s].get(dates[j], 0.0)
                      for s in set(d_long) | set(d_short) if s in rets}
                d_long = _drift_leg(d_long, jr)
                d_short = _drift_leg(d_short, jr)
        else:
            d_long, d_short = {}, {}

        # 3. Measure turnover vs drifted weights, pay costs, then size the
        #    legs off POST-cost equity (sizing pre-cost would silently lever
        #    the book up whenever costs are nonzero).
        to_long = _turnover(d_long, long_w)
        to_short = _turnover(d_short, short_w) if cfg.long_short else 0.0
        cost = (to_long + to_short) * 0.5 * equity * cfg.cost_bps / 1e4
        pre_cost_equity = equity
        equity -= cost
        total_cost += cost
        turnovers.append(to_long + to_short)

        # Dollar-neutral: 0.5x equity per leg (1.0x gross).
        long_leg = 0.5 * equity
        short_leg = 0.5 * equity if cfg.long_short else 0.0

        rebalances.append({
            "date": rdate.isoformat(),
            "n_long": len(long_w),
            "n_short": len(short_w),
            "turnover": round(to_long + to_short, 4),
            "cost": round(cost, 6),
            "equity": round(equity, 6),
        })
        prev_long_w, prev_short_w = long_w, short_w
        prev_ridx = ridx

        # 4. Hold until next rebalance (or end of sample), accruing P&L.
        #    The rebalance-day cost is folded into the first holding day's
        #    return (measured against pre-cost equity), so daily_returns
        #    compound exactly to the equity curve — costs included.
        end_idx = (dates.index(reb_dates[ri + 1]) if ri + 1 < len(reb_dates)
                   else len(dates) - 1)
        for j in range(ridx, end_idx):
            d0, d1 = dates[j], dates[j + 1]
            pnl = 0.0
            for s, w in long_w.items():
                pnl += long_leg * w * rets[s].get(d0, 0.0)
            for s, w in short_w.items():
                pnl -= short_leg * w * rets[s].get(d0, 0.0)
            # Legs drift with P&L so the next rebalance's turnover is honest.
            long_leg += sum(long_leg * w * rets[s].get(d0, 0.0)
                            for s, w in long_w.items())
            short_leg += sum(short_leg * w * rets[s].get(d0, 0.0)
                             for s, w in short_w.items())
            new_equity = equity + pnl
            base = pre_cost_equity if j == ridx else equity
            r = new_equity / base - 1.0 if base else 0.0
            equity = new_equity
            daily_returns.append(r)
            equity_curve.append(equity)
            out_dates.append(d1)

    result = BacktestResult(
        factor=cfg.factor,
        config=cfg,
        dates=out_dates,
        daily_returns=daily_returns,
        equity=equity_curve,
        rebalances=rebalances,
        avg_turnover=(sum(turnovers) / len(turnovers)) if turnovers else 0.0,
        total_cost_drag=total_cost,
    )
    result.summary = metrics_summary(daily_returns)
    result.summary["avg_turnover_per_rebalance"] = result.avg_turnover
    result.summary["total_cost_drag"] = total_cost
    return result


def run_equal_weight(price_panel: dict[str, list[tuple[date, float]]]) -> BacktestResult:
    """Buy-and-hold equal-weight benchmark over the same date range."""
    all_dates = sorted({d for series in price_panel.values() for d, _ in series})
    closes = {s: dict(series) for s, series in price_panel.items()}
    dates = [d for d in all_dates if all(d in closes[s] for s in price_panel)]
    syms = list(price_panel)
    w = 1.0 / len(syms)
    rets = {}
    for s, series in closes.items():
        sd = sorted(series.items())
        rets[s] = {sd[i][0]: sd[i + 1][1] / sd[i][1] - 1.0 for i in range(len(sd) - 1)}

    cfg = Config(factor="equal_weight", long_short=False, cost_bps=0.0)
    equity, curve, drets = 1.0, [1.0], []
    for j in range(len(dates) - 1):
        r = sum(w * rets[s].get(dates[j], 0.0) for s in syms)
        equity *= (1.0 + r)
        drets.append(r)
        curve.append(equity)
    res = BacktestResult("equal_weight", cfg, dates, drets, curve)
    res.summary = metrics_summary(drets)
    return res
