"""CLI: generate sample data, run backtests, compare factors.

Examples:
    python cli.py generate
    python cli.py run --factor momentum --cost-bps 10
    python cli.py run --factor reversal --long-only --cost-bps 25
    python cli.py compare --out results.json
"""

import argparse
import json
from pathlib import Path

from backtest import Config, run, run_equal_weight
from data_gen import load_universe, write_universe
from metrics import information_coefficient

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FACTORS = ["momentum", "reversal", "lowvol"]


def cmd_generate(args) -> None:
    write_universe(DATA)
    print(f"Generated synthetic universe in {DATA}/ (SIMULATED DATA)")


def _panel():
    try:
        return load_universe(DATA)
    except FileNotFoundError:
        print("No data/ found — generating synthetic sample data first.")
        write_universe(DATA)
        return load_universe(DATA)


def _print_summary(name: str, result) -> None:
    s = result.summary
    print(f"\n=== {name} ===")
    print(f"  factor:        {result.factor}")
    print(f"  days:          {s['n_days']}")
    print(f"  total return:  {s['total_return'] * 100:7.2f}%")
    print(f"  CAGR:          {s['cagr'] * 100:7.2f}%")
    print(f"  Sharpe:        {s['sharpe']:7.2f}")
    print(f"  Sortino:       {s['sortino']:7.2f}")
    print(f"  max drawdown:  {s['max_drawdown'] * 100:7.2f}%")
    print(f"  win rate:      {s['win_rate'] * 100:7.2f}%")
    print(f"  avg turnover:  {result.avg_turnover:7.2f} / rebalance")
    print(f"  cost drag:     {result.total_cost_drag * 100:7.2f}% of equity")


def cmd_run(args) -> None:
    panel = _panel()
    cfg = Config(factor=args.factor, cost_bps=args.cost_bps,
                 long_short=not args.long_only)
    result = run(panel, cfg)
    _print_summary(f"{args.factor} ({'long-only' if args.long_only else 'long/short'})", result)

    # Information coefficient: does the factor predict next-month returns?
    if args.ic:
        from datetime import timedelta
        sigs, fwds = [], []
        closes = {s: dict(v) for s, v in panel.items()}
        for rb in result.rebalances:
            d = rb["date"]
            from datetime import date as _d
            rd = _d.fromisoformat(d)
            from factors import compute_signals
            sig = compute_signals(panel, rd, cfg.min_history)
            for s, f in sig.items():
                if s in closes and args.factor in f:
                    # forward 21-trading-day return from rebalance date
                    series = sorted(closes[s].items())
                    idx = next(i for i, (dd, _) in enumerate(series)
                               if dd == rd)
                    if idx + 21 < len(series):
                        sigs.append(f[args.factor])
                        fwds.append(series[idx + 21][1] / series[idx][1] - 1.0)
        print(f"  avg IC:        {information_coefficient(sigs, fwds):7.3f}")


def cmd_compare(args) -> None:
    panel = _panel()
    rows = []
    for factor in FACTORS:
        result = run(panel, Config(factor=factor, cost_bps=args.cost_bps))
        _print_summary(factor, result)
        rows.append({"factor": factor, **result.summary,
                     "avg_turnover": result.avg_turnover})
    bench = run_equal_weight(panel)
    _print_summary("equal-weight benchmark", bench)
    rows.append({"factor": "equal_weight", **bench.summary})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Quant factor backtester (synthetic data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic OHLC data")
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("run", help="Run one factor backtest")
    r.add_argument("--factor", choices=FACTORS, default="momentum")
    r.add_argument("--cost-bps", type=float, default=10.0)
    r.add_argument("--long-only", action="store_true")
    r.add_argument("--ic", action="store_true",
                   help="Also report information coefficient")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="Compare all factors + benchmark")
    c.add_argument("--cost-bps", type=float, default=10.0)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
