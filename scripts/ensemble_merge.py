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


def _kemeny_consensus(orders: list[list[int]]) -> list[int]:
    """Kemeny-optimal consensus: minimize total Kendall tau distance to all inputs.

    Builds a pairwise majority matrix and solves it as a TSP using OR-Tools.
    Falls back to Borda if OR-Tools unavailable.
    """
    if not orders:
        raise ValueError("No orders provided")

    n = len(orders[0])
    pages = sorted(set(orders[0]))
    pid_to_idx = {pid: i for i, pid in enumerate(pages)}

    for o in orders:
        if len(o) != n or set(o) != set(pages):
            raise ValueError("All orders must be permutations of the same page set")

    # Build pairwise majority matrix: M[i,j] = number of orders where page i precedes page j.
    import numpy as np

    majority = np.zeros((n, n), dtype=np.float32)
    for o in orders:
        pos = {pid: r for r, pid in enumerate(o)}
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = pages[i], pages[j]
                if pos[pi] < pos[pj]:
                    majority[i, j] += 1.0
                else:
                    majority[j, i] += 1.0

    # Use majority as edge weights and solve TSP to find the ordering
    # that maximizes agreement with the majority of input orderings.
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        # Find best start: node with highest total outgoing majority.
        start = int(np.argmax(majority.sum(axis=1)))

        end = n
        n_total = n + 1
        w_max = float(np.max(majority))
        scale = 10000.0

        def cost(i: int, j: int) -> int:
            big_m = int(1e9)
            if i == end:
                return big_m
            if j == end:
                return 0
            if i == j:
                return big_m
            return int(round((w_max - float(majority[i, j])) * scale))

        # OR-Tools Python API supports either (num_nodes, num_vehicles, depot)
        # or (num_nodes, num_vehicles, starts, ends). We need distinct start/end.
        manager = pywrapcp.RoutingIndexManager(n_total, 1, [start], [end])
        routing = pywrapcp.RoutingModel(manager)

        def cb(from_index: int, to_index: int) -> int:
            return cost(manager.IndexToNode(from_index), manager.IndexToNode(to_index))

        transit_cb = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 30

        solution = routing.SolveWithParameters(params)
        if solution is None:
            return _borda_consensus(orders)

        index = routing.Start(0)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != end:
                route.append(node)
            index = solution.Value(routing.NextVar(index))

        return [pages[i] for i in route]

    except ImportError:
        return _borda_consensus(orders)


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
