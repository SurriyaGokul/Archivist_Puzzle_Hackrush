from __future__ import annotations

from dataclasses import dataclass


def inversion_count(arr: list[int]) -> int:
    """Count inversions in O(n log n) via mergesort."""

    if len(arr) < 2:
        return 0

    tmp = [0] * len(arr)

    def _sort(lo: int, hi: int) -> int:
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = _sort(lo, mid) + _sort(mid, hi)

        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                j += 1
                inv += mid - i
            k += 1
        while i < mid:
            tmp[k] = arr[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = arr[j]
            j += 1
            k += 1
        arr[lo:hi] = tmp[lo:hi]
        return inv

    return _sort(0, len(arr))


def kendall_tau_normalized_score(pred_order: list[int], true_order: list[int]) -> float:
    """Normalized Kendall Tau score in [0,1] for two permutations.

    For permutations (no ties), the hackathon score equals:
      Score = 1 - D / T
    where D is the inversion count between the two rankings.
    """

    if len(pred_order) != len(true_order):
        raise ValueError("pred_order and true_order must have same length")

    n = len(pred_order)
    if n < 2:
        return 1.0

    true_rank = {pid: i for i, pid in enumerate(true_order)}
    mapped = [true_rank[pid] for pid in pred_order]
    d = inversion_count(mapped)
    t = n * (n - 1) // 2
    return 1.0 - (d / t)


@dataclass(frozen=True)
class EvalResult:
    score: float
    n_pages: int
