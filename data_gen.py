"""Deterministic synthetic OHLC data generator.

Every number in data/ is invented. The generator is seeded, so the same seed
always produces the same CSVs — the demo is reproducible anywhere.

SIMULATED DATA — nothing here is real market data.
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

BANNER = (
    "SIMULATED DATA — synthetically generated for demo purposes only. "
    "Not real market data. Do not use for real trading decisions."
)

# Fictional companies. Tickers are invented.
UNIVERSE = [
    ("NIMX", "Nimbus Labs"),
    ("VLTQ", "Voltaic Systems"),
    ("HLIO", "Helios Home"),
    ("QNTM", "Quantia Robotics"),
    ("BRCK", "Brickline Materials"),
    ("SLVR", "Silverline Foods"),
    ("CRBN", "Carbon & Co"),
    ("PLSR", "Pulsar Networks"),
    ("DRFT", "Driftwell Logistics"),
    ("MSSA", "Mesa Semiconductors"),
    ("TNDR", "Tundra Energy"),
    ("FRST", "Frostline Apparel"),
]


def business_days(start: date, n: int) -> list[date]:
    """First n business days starting at `start` (inclusive if weekday)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate_symbol(symbol: str, name: str, start: date, n_days: int,
                    seed: int) -> list[dict]:
    """Generate one synthetic OHLC series.

    Geometric Brownian motion with regime switching (bull/bear/flat blocks)
    so factors have something real to chew on: trends persist, then break.
    Fully deterministic for a given seed.
    """
    rng = random.Random(seed)
    dates = business_days(start, n_days)

    # Regime blocks of ~126 trading days (~6 months): drift + vol per block.
    regimes = []
    r2 = random.Random(seed + 999)
    n_blocks = math.ceil(n_days / 126)
    for _ in range(n_blocks):
        kind = r2.random()
        if kind < 0.45:
            regimes.append((0.0009, 0.016))   # bull: positive drift
        elif kind < 0.70:
            regimes.append((-0.0007, 0.024))  # bear: negative drift, higher vol
        else:
            regimes.append((0.0001, 0.014))   # flat: chop

    rows: list[dict] = []
    price = 40.0 + rng.random() * 160.0  # invented starting price
    for i, d in enumerate(dates):
        drift, vol = regimes[i // 126]
        shock = rng.gauss(0.0, 1.0)
        daily_ret = drift + vol * shock
        open_p = price
        close_p = price * (1.0 + daily_ret)
        # Intraday range around the open->close path (invented, small).
        span = abs(close_p - open_p) + open_p * vol * 0.4 * rng.random()
        high_p = max(open_p, close_p) + span * rng.random() * 0.5
        low_p = min(open_p, close_p) - span * rng.random() * 0.5
        volume = int(200_000 + rng.random() * 1_800_000)
        rows.append({
            "date": d.isoformat(),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": volume,
        })
        price = close_p
    return rows


def write_universe(data_dir: Path, start: date = date(2020, 1, 2),
                   n_days: int = 1008, seed: int = 7) -> None:
    """Write one CSV per symbol plus a universe.csv manifest."""
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (symbol, name) in enumerate(UNIVERSE):
        rows = generate_symbol(symbol, name, start, n_days, seed + i * 101)
        path = data_dir / f"{symbol}.csv"
        with path.open("w", newline="") as f:
            f.write(f"# {BANNER}\n")
            f.write(f"# symbol: {symbol}\n")
            f.write(f"# name: {name}\n")
            writer = csv.DictWriter(
                f, fieldnames=["date", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            writer.writerows(rows)
        manifest.append({"symbol": symbol, "name": name, "seed": seed + i * 101})
    with (data_dir / "universe.csv").open("w", newline="") as f:
        f.write(f"# {BANNER}\n")
        writer = csv.DictWriter(f, fieldnames=["symbol", "name", "seed"])
        writer.writeheader()
        writer.writerows(manifest)


def load_universe(data_dir: Path) -> dict[str, list[tuple[date, float]]]:
    """Load data/ into {symbol: [(date, close), ...]} sorted by date.

    Skips `#` comment lines. Raises FileNotFoundError if data/ is missing.
    """
    panel: dict[str, list[tuple[date, float]]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name == "universe.csv":
            continue
        symbol = csv_path.stem
        with csv_path.open() as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        reader = csv.DictReader(lines)
        series = [
            (date.fromisoformat(r["date"]), float(r["close"])) for r in reader
        ]
        series.sort(key=lambda x: x[0])
        panel[symbol] = series
    if not panel:
        raise FileNotFoundError(f"No symbol CSVs found in {data_dir}")
    return panel


if __name__ == "__main__":
    write_universe(Path(__file__).parent / "data")
    print(f"Wrote {len(UNIVERSE)} synthetic symbols to data/")
