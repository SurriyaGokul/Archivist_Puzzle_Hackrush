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
    p.add_argument("--method", choices=["borda", "kemeny"], default="kemeny", help="Consensus method")
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


def _kendall_inversions(order_a: list[int], order_b: list[int]) -> int:
    """Return Kendall inversion count between two permutations.

    Both lists must contain the same items exactly once.
    """

    if len(order_a) != len(order_b):
        raise ValueError("Orders must have the same length")

    n = len(order_a)
    pos = {pid: i for i, pid in enumerate(order_b)}
    try:
        arr = [pos[pid] for pid in order_a]
    except KeyError as e:
        raise ValueError("Orders must contain the same page ids") from e

    # Fenwick tree over positions 0..n-1.
    bit = [0] * (n + 1)

    def bit_add(i: int, delta: int) -> None:
        while i <= n:
            bit[i] += delta
            i += i & -i

    def bit_sum(i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    inv = 0
    seen = 0
    for x in arr:
        i = x + 1
        leq = bit_sum(i)
        inv += seen - leq
        bit_add(i, 1)
        seen += 1

    return inv


def _total_kendall_inversions(candidate: list[int], orders: list[list[int]]) -> int:
    return sum(_kendall_inversions(candidate, o) for o in orders)


def _kemeny_insert_heuristic(orders: list[list[int]]) -> list[int]:
    """Heuristic for (approx) Kemeny median using pairwise preferences.

    - Build pairwise matrix M[a,b] = #orders where a precedes b.
    - Construct a permutation by inserting items (Borda seed) at the position
      that maximizes incremental agreement.
    - Apply a few adjacent-swap improvement passes.
    """

    if not orders:
        raise ValueError("No orders provided")

    n = len(orders[0])
    pages = sorted(set(orders[0]))
    if len(pages) != n:
        raise ValueError("Orders must be permutations (no duplicates)")

    for o in orders:
        if len(o) != n or set(o) != set(pages):
            raise ValueError("All orders must be permutations of the same page set")

    pid_to_idx = {pid: i for i, pid in enumerate(pages)}

    import numpy as np

    majority = np.zeros((n, n), dtype=np.int16)
    for o in orders:
        pos = np.empty(n, dtype=np.int32)
        for r, pid in enumerate(o):
            pos[pid_to_idx[pid]] = r
        for i in range(n):
            for j in range(i + 1, n):
                if pos[i] < pos[j]:
                    majority[i, j] += 1
                else:
                    majority[j, i] += 1

    seed = _borda_consensus(orders)
    perm_idx: list[int] = []
    for pid in seed:
        x = pid_to_idx[pid]

        # Score for inserting x at each position k.
        # k=0 => x before all existing => sum majority[x, y]
        after = int(sum(int(majority[x, y]) for y in perm_idx))
        before = 0
        best_score = before + after
        best_k = 0

        for k, y in enumerate(perm_idx):
            before += int(majority[y, x])
            after -= int(majority[x, y])
            score = before + after
            if score > best_score:
                best_score = score
                best_k = k + 1

        perm_idx.insert(best_k, x)

    # A few adjacent-swap improvement passes.
    for _ in range(min(2 * n, 500)):
        swapped = False
        for k in range(n - 1):
            a = perm_idx[k]
            b = perm_idx[k + 1]
            if int(majority[b, a]) > int(majority[a, b]):
                perm_idx[k], perm_idx[k + 1] = b, a
                swapped = True
        if not swapped:
            break

    return [pages[i] for i in perm_idx]


def _kemeny_consensus(orders: list[list[int]]) -> list[int]:
    """Approximate Kemeny consensus (minimize total Kendall distance).

    NOTE: A previous OR-Tools/TSP formulation was fragile (API differences) and
    can produce very poor consensus because it only scores *adjacent* pairs.
    This implementation uses a safer, pairwise-preference heuristic and then
    selects the best candidate by directly minimizing total Kendall inversions
    to the input permutations.
    """

    if not orders:
        raise ValueError("No orders provided")

    if len(orders) == 1:
        return list(orders[0])

    # For exactly two inputs, Borda/average-rank is already a Kemeny-optimal median
    # (it preserves all pairwise agreements between the two permutations).
    borda = _borda_consensus(orders)
    if len(orders) == 2:
        return borda

    # Heuristic candidate + robust selection among safe candidates.
    heuristic = _kemeny_insert_heuristic(orders)
    candidates: list[list[int]] = [heuristic, borda, *[list(o) for o in orders]]

    best = min(candidates, key=lambda c: _total_kendall_inversions(c, orders))
    return best


def main() -> None:
    args = _parse_args()
    in_dirs = [Path(x.strip()) for x in args.in_dirs.split(",") if x.strip()]
    if not in_dirs:
        raise ValueError("--in_dirs must include at least one directory")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus_fn = _kemeny_consensus if args.method == "kemeny" else _borda_consensus

    for book in ["BookA", "BookB"]:
        orders: list[list[int]] = []
        for d in in_dirs:
            csv_path = d / f"{book}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            orders.append(_read_order(csv_path))

        merged = consensus_fn(orders)
        write_submission(merged, out_dir / f"{book}.csv")
        print(f"wrote {out_dir / f'{book}.csv'}")


if __name__ == "__main__":
    main()
