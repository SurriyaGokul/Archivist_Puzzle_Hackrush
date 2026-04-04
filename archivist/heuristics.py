from __future__ import annotations

import numpy as np

from archivist.data import normalize_text


def boundary_overlap_matrix(
    texts: list[str],
    *,
    overlap_words: int = 80,
    max_overlap: int = 12,
    lowercase: bool = True,
) -> np.ndarray:
    """Compute a cheap directed adjacency hint via exact boundary word overlap.

    W[i, j] = k where k is the longest (up to `max_overlap`) exact word overlap:
      suffix(words_i, k) == prefix(words_j, k)

    This often catches hard continuations (e.g. hyphenation artifacts removed by
    normalization, repeated phrases across page boundaries, dialogue carry-over).

    Notes:
    - Diagonal is 0.0 (LM scorer already sets diagonal to -inf).
    - Intended to be added as: W_total = W_lm + overlap_weight * W_overlap
    """

    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    overlap_words = max(1, int(overlap_words))
    max_overlap = max(1, int(max_overlap))

    prefixes: list[list[str]] = []
    suffixes: list[list[str]] = []

    for t in texts:
        t = normalize_text(t)
        if lowercase:
            t = t.lower()
        words = t.split()
        prefixes.append(words[:overlap_words])
        suffixes.append(words[-overlap_words:])

    w = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        s = suffixes[i]
        if not s:
            continue
        for j in range(n):
            if i == j:
                continue
            p = prefixes[j]
            if not p:
                continue
            kmax = min(max_overlap, len(s), len(p))
            best = 0
            for k in range(kmax, 0, -1):
                if s[-k:] == p[:k]:
                    best = k
                    break
            if best:
                w[i, j] = float(best)

    return w
