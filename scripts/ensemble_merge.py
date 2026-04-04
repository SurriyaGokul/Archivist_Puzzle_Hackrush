from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archivist.solver import write_submission


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge multiple candidate submissions into a single consensus submission "
            "(Borda/average-rank)."
        )
    )
    p.add_argument(
        "--in_dirs",
        required=True,
        help="Comma-separated directories, each containing BookA.csv and BookB.csv",
    )
    p.add_argument("--out_dir", required=True, help="Directory to write merged BookA.csv and BookB.csv")
    return p.parse_args()


def _read_order(csv_path: Path) -> list[int]:
    df = pd.read_csv(csv_path)
    if "original_page" not in df.columns or "shuffled_page" not in df.columns:
        raise ValueError(f"Invalid submission format: {csv_path}")

    n = len(df)
    orig = df["original_page"].tolist()
    if orig != list(range(1, n + 1)):
        raise ValueError(f"original_page must be 1..N in order: {csv_path}")

    return [int(x) for x in df["shuffled_page"].tolist()]


def _borda_consensus(orders: list[list[int]]) -> list[int]:
    if not orders:
        raise ValueError("No orders provided")

    n = len(orders[0])
    pages = set(orders[0])
    for o in orders:
        if len(o) != n:
            raise ValueError("All orders must have the same length")
        if set(o) != pages:
            raise ValueError("All orders must contain the same page ids")

    # Sum of ranks (lower is better).
    rank_sum: dict[int, int] = {pid: 0 for pid in pages}
    for o in orders:
        for r, pid in enumerate(o):
            rank_sum[pid] += r

    return sorted(pages, key=lambda pid: (rank_sum[pid], pid))


def main() -> None:
    args = _parse_args()
    in_dirs = [Path(x.strip()) for x in args.in_dirs.split(",") if x.strip()]
    if not in_dirs:
        raise ValueError("--in_dirs must include at least one directory")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for book in ["BookA", "BookB"]:
        orders: list[list[int]] = []
        for d in in_dirs:
            csv_path = d / f"{book}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            orders.append(_read_order(csv_path))

        merged = _borda_consensus(orders)
        write_submission(merged, out_dir / f"{book}.csv")
        print(f"wrote {out_dir / f'{book}.csv'}")


if __name__ == "__main__":
    main()
