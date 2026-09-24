"""Performance and risk metrics for backtest results. Pure functions."""

import math
import statistics


def equity_curve(daily_returns: list[float], start: float = 1.0) -> list[float]:
    eq = [start]
    for r in daily_returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def max_drawdown(equity: list[float]) -> dict:
    """Returns {max_dd, peak_idx, trough_idx} as fractions (0.2 = -20%)."""
    peak = equity[0]
    peak_idx = 0
    worst = 0.0
    w_peak = 0
    w_trough = 0
    for i, v in enumerate(equity):
        if v > peak:
            peak = v
            peak_idx = i
        dd = (peak - v) / peak if peak else 0.0
        if dd > worst:
            worst = dd
            w_peak = peak_idx
            w_trough = i
    return {"max_dd": worst, "peak_idx": w_peak, "trough_idx": w_trough}


def sharpe_ratio(daily_returns: list[float], periods: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    sd = statistics.pstdev(daily_returns)
    if sd == 0:
        return 0.0
    return statistics.fmean(daily_returns) / sd * math.sqrt(periods)


def sortino_ratio(daily_returns: list[float], periods: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    downside = [min(0.0, r) for r in daily_returns]
    dd = statistics.pstdev(downside)
    if dd == 0:
        return 0.0
    return statistics.fmean(daily_returns) / dd * math.sqrt(periods)


def summary(daily_returns: list[float]) -> dict:
    """One dict with the headline numbers. daily_returns: net of costs."""
    eq = equity_curve(daily_returns)
    total = eq[-1] / eq[0] - 1.0
    years = len(daily_returns) / 252
    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    dd = max_drawdown(eq)
    wins = sum(1 for r in daily_returns if r > 0)
    return {
        "n_days": len(daily_returns),
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe_ratio(daily_returns),
        "sortino": sortino_ratio(daily_returns),
        "max_drawdown": dd["max_dd"],
        "win_rate": wins / len(daily_returns) if daily_returns else 0.0,
        "best_day": max(daily_returns) if daily_returns else 0.0,
        "worst_day": min(daily_returns) if daily_returns else 0.0,
    }


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks for ties (proper Spearman). Constant input -> all equal."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def information_coefficient(signals: list[float],
                            forward_returns: list[float]) -> float:
    """Spearman rank correlation between signal and next-period return.

    The IC measures whether the factor actually predicts returns — the
    honest version of "does this factor work", independent of portfolio
    construction choices.
    """
    if len(signals) != len(forward_returns) or len(signals) < 3:
        return 0.0
    rs, rf = _ranks(signals), _ranks(forward_returns)
    n = len(rs)
    mr, mf = statistics.fmean(rs), statistics.fmean(rf)
    cov = sum((a - mr) * (b - mf) for a, b in zip(rs, rf)) / n
    sr, sf = statistics.pstdev(rs), statistics.pstdev(rf)
    if sr == 0 or sf == 0:
        return 0.0
    return cov / (sr * sf)
