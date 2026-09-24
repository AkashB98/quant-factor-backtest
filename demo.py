"""One-command demo: generate data, run all factors, print comparison table."""

from pathlib import Path

from backtest import Config, run, run_equal_weight
from data_gen import load_universe, write_universe

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main() -> None:
    try:
        panel = load_universe(DATA)
    except FileNotFoundError:
        print("Generating synthetic sample data...")
        write_universe(DATA)
        panel = load_universe(DATA)

    print(f"\nUniverse: {len(panel)} synthetic symbols "
          f"(SIMULATED DATA — not real market data)\n")
    print(f"{'strategy':<22}{'total':>9}{'CAGR':>9}{'Sharpe':>8}"
          f"{'maxDD':>8}{'turnover':>10}")
    print("-" * 70)
    for factor in ["momentum", "reversal", "lowvol"]:
        r = run(panel, Config(factor=factor, cost_bps=10.0))
        s = r.summary
        print(f"{factor + ' L/S':<22}{s['total_return'] * 100:8.1f}%"
              f"{s['cagr'] * 100:8.1f}%{s['sharpe']:8.2f}"
              f"{s['max_drawdown'] * 100:7.1f}%{r.avg_turnover:10.2f}")
    b = run_equal_weight(panel)
    s = b.summary
    print(f"{'equal-weight bench':<22}{s['total_return'] * 100:8.1f}%"
          f"{s['cagr'] * 100:8.1f}%{s['sharpe']:8.2f}"
          f"{s['max_drawdown'] * 100:7.1f}%")
    print("\nWalk-forward, monthly rebalance, 10 bps/trade, dollar-neutral L/S.")
    print("Run `python cli.py run --factor momentum --ic` for detail,")
    print("or `python server.py` for charts in your browser.")


if __name__ == "__main__":
    main()
