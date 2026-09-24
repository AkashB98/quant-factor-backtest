"""Cross-sectional equity factors, computed point-in-time.

The cardinal rule of backtesting: a signal used on date T may only touch
prices with date <= T. `compute_signals` enforces this by construction — it
truncates every series at `asof` BEFORE any math happens, so future data
cannot leak in even by accident. There is a regression test that proves it
(tests/test_backtest.py::TestNoLookahead).

Factors (all standard academic definitions):
- momentum: 12-month return skipping the most recent month (12-1).
- reversal: prior-month return, negated (short-term reversal).
- lowvol: trailing volatility, negated (low-volatility anomaly).
"""

import statistics
from datetime import date

MIN_HISTORY = 260  # need >= 252 trading days for the longest lookback


def _pct_changes(closes: list[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(closes, closes[1:])]


def momentum(closes: list[float], lookback: int = 252, skip: int = 21) -> float | None:
    """(P_{t-skip} / P_{t-lookback}) - 1. None if not enough history."""
    if len(closes) < lookback + 1:
        return None
    return closes[-1 - skip] / closes[-1 - lookback] - 1.0


def reversal(closes: list[float], lookback: int = 21) -> float | None:
    """-(P_t / P_{t-lookback} - 1). None if not enough history."""
    if len(closes) < lookback + 1:
        return None
    return -(closes[-1] / closes[-1 - lookback] - 1.0)


def lowvol(closes: list[float], lookback: int = 63) -> float | None:
    """-(trailing stdev of daily returns). None if not enough history."""
    if len(closes) < lookback + 1:
        return None
    rets = _pct_changes(closes[-(lookback + 1):])
    if len(set(rets)) < 2:
        return None
    return -statistics.pstdev(rets)


FACTOR_FNS = {
    "momentum": momentum,
    "reversal": reversal,
    "lowvol": lowvol,
}


def compute_signals(price_panel: dict[str, list[tuple[date, float]]],
                    asof: date,
                    min_history: int = MIN_HISTORY) -> dict[str, dict[str, float]]:
    """Cross-sectional factor values using only data with date <= asof.

    price_panel: {symbol: [(date, close), ...]} (any length, sorted or not).
    Returns {symbol: {"momentum": ..., "reversal": ..., "lowvol": ...}} for
    symbols with at least `min_history` closes. Symbols lacking history are
    skipped.
    """
    out: dict[str, dict[str, float]] = {}
    for symbol, series in price_panel.items():
        closes = [c for d, c in series if d <= asof]
        if len(closes) < min_history:
            continue
        vals: dict[str, float] = {}
        for fname, fn in FACTOR_FNS.items():
            v = fn(closes)
            if v is not None:
                vals[fname] = v
        if len(vals) == len(FACTOR_FNS):
            out[symbol] = vals
    return out
