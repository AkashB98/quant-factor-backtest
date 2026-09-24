"""Hermetic unit tests: deterministic, no network, no clock, no randomness."""

import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import (Config, _drift_leg, _turnover, month_end_dates, run,
                      target_weights)
from data_gen import (business_days, generate_symbol, load_universe,
                      write_universe)
from factors import (compute_signals, lowvol, momentum, reversal)
from metrics import (equity_curve, information_coefficient, max_drawdown,
                     sharpe_ratio, summary)


def tiny_panel(n_days=400, n_syms=6, seed=11):
    panel = {}
    for i in range(n_syms):
        rows = generate_symbol(f"S{i}", f"Synthetic {i}",
                               date(2020, 1, 2), n_days, seed + i * 101)
        panel[f"S{i}"] = [(date.fromisoformat(r["date"]), r["close"])
                          for r in rows]
    return panel


class TestDataGen(unittest.TestCase):
    def test_deterministic(self):
        a = generate_symbol("X", "X", date(2020, 1, 2), 50, 5)
        b = generate_symbol("X", "X", date(2020, 1, 2), 50, 5)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = generate_symbol("X", "X", date(2020, 1, 2), 50, 5)
        b = generate_symbol("X", "X", date(2020, 1, 2), 50, 6)
        self.assertNotEqual(a, b)

    def test_business_days_weekdays_only(self):
        days = business_days(date(2020, 1, 1), 10)
        self.assertTrue(all(d.weekday() < 5 for d in days))
        self.assertEqual(len(days), 10)

    def test_ohlc_sanity(self):
        rows = generate_symbol("X", "X", date(2020, 1, 2), 50, 5)
        for r in rows:
            self.assertLessEqual(r["low"], r["open"])
            self.assertLessEqual(r["low"], r["close"])
            self.assertGreaterEqual(r["high"], r["open"])
            self.assertGreaterEqual(r["high"], r["close"])
            self.assertGreater(r["close"], 0)

    def test_write_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_universe(d, n_days=60)
            panel = load_universe(d)
            self.assertEqual(len(panel), 12)
            for series in panel.values():
                self.assertEqual(len(series), 60)
                self.assertTrue(all(isinstance(x[0], date) for x in series))

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_universe(Path(td) / "nope")


class TestFactors(unittest.TestCase):
    def test_momentum_handcalc(self):
        # price doubles over the lookback window (skip=0 for simplicity)
        closes = [100.0] * 253
        closes[-1] = 200.0
        closes[-2] = 200.0  # ensure P_{t-skip} is 200
        self.assertAlmostEqual(momentum(closes, lookback=252, skip=0), 1.0)

    def test_momentum_none_when_short(self):
        self.assertIsNone(momentum([100.0] * 100))

    def test_reversal_sign(self):
        rising = [100.0 + i for i in range(30)]
        self.assertLess(reversal(rising), 0)  # up last month -> negative

    def test_lowvol_prefers_calm(self):
        calm = [100.0 * (1 + 0.001 * ((i % 2) * 2 - 1)) for i in range(70)]
        wild = [100.0 * (1 + 0.05 * ((i % 2) * 2 - 1)) for i in range(70)]
        self.assertGreater(lowvol(calm), lowvol(wild))

    def test_skips_young_symbols(self):
        panel = tiny_panel(n_days=400, n_syms=3)
        # add one symbol with only 10 days of history
        panel["BABY"] = panel["S0"][:10]
        asof = panel["S0"][300][0]
        sigs = compute_signals(panel, asof)
        self.assertNotIn("BABY", sigs)
        self.assertIn("S0", sigs)


class TestNoLookahead(unittest.TestCase):
    """The cardinal backtest rule: future prices must not affect signals."""

    def test_future_corruption_changes_nothing(self):
        panel = tiny_panel(n_days=400, n_syms=6)
        asof = panel["S0"][300][0]
        clean = compute_signals(panel, asof)

        corrupted = {}
        for s, series in panel.items():
            corrupted[s] = [(d, c * 1000.0 if d > asof else c)
                            for d, c in series]
        dirty = compute_signals(corrupted, asof)
        self.assertEqual(clean, dirty)

    def test_truncated_panel_matches(self):
        panel = tiny_panel(n_days=400, n_syms=6)
        asof = panel["S0"][300][0]
        full = compute_signals(panel, asof)
        trunc = {s: [(d, c) for d, c in series if d <= asof]
                 for s, series in panel.items()}
        # truncated panel with a far-future asof must give identical signals
        part = compute_signals(trunc, date(2099, 1, 1))
        self.assertEqual(full, part)

    def test_run_ignores_future(self):
        panel = tiny_panel(n_days=400, n_syms=6)
        cfg = Config(factor="reversal", cost_bps=0.0)
        r1 = run(panel, cfg)
        # corrupt the LAST 50 days of every series: earlier windows unaffected
        corrupted = {}
        for s, series in panel.items():
            corrupted[s] = [(d, c * 5.0 if i >= len(series) - 50 else c)
                            for i, (d, c) in enumerate(series)]
        r2 = run(corrupted, cfg)
        # daily returns before the corruption window must be identical
        n = len(r1.daily_returns)
        self.assertEqual(r1.daily_returns[:n - 60], r2.daily_returns[:n - 60])


class TestEngine(unittest.TestCase):
    def test_month_end_dates(self):
        days = business_days(date(2021, 1, 1), 90)
        ends = month_end_dates(days)
        months = {(d.year, d.month) for d in ends}
        self.assertEqual(len(ends), len(months))  # one per month
        self.assertTrue(all(
            ends[i] < ends[i + 1] for i in range(len(ends) - 1)))

    def test_weights_sum_to_one_per_leg(self):
        panel = tiny_panel()
        asof = panel["S0"][300][0]
        sigs = compute_signals(panel, asof)
        long_w, short_w = target_weights(sigs, "momentum", Config())
        self.assertAlmostEqual(sum(long_w.values()), 1.0)
        self.assertAlmostEqual(sum(short_w.values()), 1.0)
        self.assertTrue(set(long_w).isdisjoint(set(short_w)))

    def test_long_only_has_no_shorts(self):
        panel = tiny_panel()
        asof = panel["S0"][300][0]
        sigs = compute_signals(panel, asof)
        long_w, short_w = target_weights(sigs, "momentum",
                                         Config(long_short=False))
        self.assertEqual(short_w, {})
        self.assertAlmostEqual(sum(long_w.values()), 1.0)

    def test_costs_reduce_returns(self):
        panel = tiny_panel()
        free = run(panel, Config(factor="reversal", cost_bps=0.0))
        costly = run(panel, Config(factor="reversal", cost_bps=300.0))
        self.assertGreater(len(free.daily_returns), 0)
        self.assertLess(costly.summary["total_return"],
                        free.summary["total_return"])
        self.assertGreater(costly.total_cost_drag, 0)

    def test_walkforward_covers_sample(self):
        panel = tiny_panel()
        r = run(panel, Config(factor="momentum"))
        self.assertEqual(len(r.daily_returns), len(r.dates) - 1)
        self.assertTrue(all(r.dates[i] < r.dates[i + 1]
                            for i in range(len(r.dates) - 1)))
        self.assertEqual(len(set(r.dates)), len(r.dates))
        # equity curve is consistent with daily returns
        self.assertAlmostEqual(r.equity[-1],
                               equity_curve(r.daily_returns)[-1], places=9)

    def test_rebalances_logged(self):
        panel = tiny_panel()
        r = run(panel, Config(factor="lowvol"))
        self.assertGreater(len(r.rebalances), 1)
        for rb in r.rebalances:
            self.assertGreaterEqual(rb["n_long"], 1)

    def test_turnover_zero_when_unchanged(self):
        w = {"A": 0.5, "B": 0.5}
        self.assertEqual(_turnover(w, dict(w)), 0.0)
        self.assertAlmostEqual(_turnover({"A": 1.0}, {"B": 1.0}), 2.0)

    def test_drift_leg_renormalizes(self):
        d = _drift_leg({"A": 0.5, "B": 0.5}, {"A": 0.1, "B": -0.1})
        self.assertAlmostEqual(sum(d.values()), 1.0)
        self.assertGreater(d["A"], d["B"])

    def test_needs_enough_history(self):
        panel = tiny_panel(n_days=100)
        with self.assertRaises(ValueError):
            run(panel, Config())


class TestMetrics(unittest.TestCase):
    def test_sharpe_zero_vol(self):
        self.assertEqual(sharpe_ratio([0.01] * 50), 0.0)

    def test_sharpe_positive_for_good_returns(self):
        self.assertGreater(sharpe_ratio([0.01, 0.02, 0.015, 0.005] * 25), 0)

    def test_max_drawdown_handcalc(self):
        dd = max_drawdown([1.0, 0.9, 0.95, 0.8, 0.85])
        self.assertAlmostEqual(dd["max_dd"], 0.2)

    def test_summary_keys(self):
        s = summary([0.01, -0.005, 0.02])
        for k in ("total_return", "cagr", "sharpe", "sortino",
                  "max_drawdown", "win_rate", "best_day", "worst_day"):
            self.assertIn(k, s)

    def test_ic_perfect(self):
        self.assertAlmostEqual(
            information_coefficient([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(
            information_coefficient([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(information_coefficient([1, 1, 1], [1, 2, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
