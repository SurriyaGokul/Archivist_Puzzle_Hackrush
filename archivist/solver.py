from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from archivist.anchors import extract_anchors
from archivist.cache import SQLiteScoreCache
from archivist.config import SolverConfig
from archivist.data import normalize_text, word_window
from archivist.embeddings import Embedder, cosine_sim_matrix, spectral_seriation
from archivist.heuristics import boundary_overlap_matrix
from archivist.lm import LMConfig, LMScorer
from archivist.types import Anchor, Page


def solve_pages(pages: list[Page], *, config: SolverConfig, cache_dir: str | Path = ".cache") -> list[int]:
    """Return predicted page_id order (length N) for a single book."""

    id_to_idx = {p.page_id: i for i, p in enumerate(pages)}

    anchors = extract_anchors(pages)
    anchors_sorted = sorted(anchors, key=lambda a: a.chapter_num)

    # Precompute windows.
    head_text = []
    tail_text = []
    full_text = []
    for p in pages:
        full = normalize_text(p.text)
        h, t = word_window(full, head_words=config.head_words, tail_words=config.tail_words)
        full_text.append(full)
        head_text.append(h)
        tail_text.append(t)

    # Embeddings (used for chapter bucketing + edge features).
    embedder = Embedder(model_name=config.embed_model, batch_size=config.embed_batch_size)
    full_emb = embedder.encode(full_text)

    head_emb = embedder.encode(head_text)
    tail_emb = embedder.encode(tail_text)

    # Bucketing.
    if anchors_sorted:
        buckets = _assign_to_chapters_spectral(
            pages,
            anchors_sorted,
            full_emb,
            method=config.assign_method,
            knn=config.spectral_knn,
        )
        ordered_chapter_keys: list[int | str] = ["__pre__", *[a.chapter_num for a in anchors_sorted], "__post__"]
    else:
        # No anchors: single bucket.
        buckets = {"__all__": [p.page_id for p in pages]}
        ordered_chapter_keys = ["__all__"]

    # LM scorers.
    scorers: list[LMScorer] = []
    if config.lm_models:
        for m in config.lm_models:
            cfg = LMConfig(
                model_name=m,
                load_in_4bit=config.lm_load_in_4bit,
                load_in_8bit=config.lm_load_in_8bit,
                dtype=config.lm_dtype,
                device=config.lm_device,
                max_length=config.lm_max_length,
            )
            scorers.append(LMScorer(cfg))

    cache = SQLiteScoreCache(Path(cache_dir) / "scores.sqlite")

    # Solve buckets in order and concatenate.
    out_order: list[int] = []
    for key in ordered_chapter_keys:
        page_ids = buckets.get(key, [])
        if not page_ids:
            continue

        if isinstance(key, int):
            anchor_page_id = next((a.page_id for a in anchors_sorted if a.chapter_num == key), None)
        else:
            anchor_page_id = None

        seq = _order_bucket(
            page_ids,
            anchor_page_id=anchor_page_id,
            id_to_idx=id_to_idx,
            head_text=head_text,
            tail_text=tail_text,
            head_emb=head_emb,
            tail_emb=tail_emb,
            full_text=full_text,
            config=config,
            scorers=scorers,
            cache=cache,
        )
        out_order.extend(seq)

    cache.close()

    # Final sanity: permutation of input page ids.
    if set(out_order) != {p.page_id for p in pages} or len(out_order) != len(pages):
        raise RuntimeError("Solver produced invalid permutation")

    return out_order


def _assign_to_chapters_spectral(
    pages: list[Page],
    anchors: list[Anchor],
    full_emb: np.ndarray,
    *,
    method: str,
    knn: int,
) -> dict[int | str, list[int]]:
    """Assign each page to a chapter bucket.

    Returns buckets keyed by chapter_num plus '__pre__'/'__post__'.
    """

    buckets: dict[int | str, list[int]] = {"__pre__": [], "__post__": []}
    for a in anchors:
        buckets[a.chapter_num] = []

    if method not in {"spectral", "nearest_anchor"}:
        method = "spectral"

    if method == "nearest_anchor":
        # Nearest anchor by cosine similarity in full-text embedding space.
        page_ids = [p.page_id for p in pages]
        id_to_pos = {pid: i for i, pid in enumerate(page_ids)}
        anchor_ids = [a.page_id for a in anchors]
        anchor_pos = [id_to_pos[pid] for pid in anchor_ids]

        sim = full_emb @ full_emb[anchor_pos].T
        for i, pid in enumerate(page_ids):
            if pid in anchor_ids:
                # Put anchor in its own chapter.
                chap = next(a.chapter_num for a in anchors if a.page_id == pid)
                buckets[chap].append(pid)
                continue
            best = int(np.argmax(sim[i]))
            chap = anchors[best].chapter_num
            buckets[chap].append(pid)
        return buckets

    # Spectral coordinate.
    _, coord = spectral_seriation(full_emb, knn=knn)

    anchor_ids = [a.page_id for a in anchors]
    id_to_coord = {p.page_id: float(coord[i]) for i, p in enumerate(pages)}

    anchor_coords = np.array([id_to_coord[a.page_id] for a in anchors], dtype=np.float32)

    # Make coordinates increase with chapter index (stable orientation).
    if len(anchor_coords) >= 2:
        x = np.arange(len(anchor_coords), dtype=np.float32)
        corr = float(np.corrcoef(x, anchor_coords)[0, 1])
        if np.isfinite(corr) and corr < 0:
            coord = -coord
            id_to_coord = {p.page_id: float(coord[i]) for i, p in enumerate(pages)}
            anchor_coords = -anchor_coords

    # Enforce monotone increasing anchor coords using isotonic regression (if available).
    try:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        anchor_fit = ir.fit_transform(np.arange(len(anchor_coords)), anchor_coords)
    except Exception:
        anchor_fit = anchor_coords

    anchor_fit_list = [float(x) for x in anchor_fit]

    for p in pages:
        c = id_to_coord[p.page_id]
        if p.page_id in anchor_ids:
            chap = next(a.chapter_num for a in anchors if a.page_id == p.page_id)
            buckets[chap].append(p.page_id)
            continue
        # Assign by chapter starts: pages between Chapter N and N+1 belong to Chapter N.
        k = bisect.bisect_right(anchor_fit_list, c) - 1
        if k < 0:
            buckets["__pre__"].append(p.page_id)
        else:
            chap = anchors[min(k, len(anchors) - 1)].chapter_num
            buckets[chap].append(p.page_id)

    return buckets


def _order_bucket(
    page_ids: list[int],
    *,
    anchor_page_id: int | None,
    id_to_idx: dict[int, int],
    head_text: list[str],
    tail_text: list[str],
    head_emb: np.ndarray,
    tail_emb: np.ndarray,
    full_text: list[str],
    config: SolverConfig,
    scorers: list[LMScorer],
    cache: SQLiteScoreCache,
) -> list[int]:
    # If bucket is trivial.
    if len(page_ids) == 1:
        return page_ids

    # Build local index mapping.
    local_ids = list(dict.fromkeys(page_ids))

    if anchor_page_id is not None and anchor_page_id in local_ids:
        # Ensure anchor is first in local ordering.
        local_ids.remove(anchor_page_id)
        local_ids = [anchor_page_id, *local_ids]
        start_local = 0
    else:
        start_local = 0

    n = len(local_ids)

    # Similarity feature (tail->head).
    local_idx = [id_to_idx[pid] for pid in local_ids]
    sim = cosine_sim_matrix(tail_emb[local_idx], head_emb[local_idx]).astype(np.float32)
    np.fill_diagonal(sim, -1.0)

    # Boundary overlap feature.
    texts = [full_text[i] for i in local_idx]
    overlap = boundary_overlap_matrix(
        texts,
        overlap_words=config.overlap_words,
        max_overlap=config.max_overlap,
        lowercase=True,
    )
    if config.max_overlap > 0:
        overlap_norm = overlap / float(config.max_overlap)
    else:
        overlap_norm = overlap

    cheap = (config.w_emb * sim + config.w_overlap * overlap_norm).astype(np.float32)

    # LM feature (optional).
    lm_scores = np.zeros((n, n), dtype=np.float32)
    if scorers:
        import hashlib

        def edge_key(prefix: str, separator: str, target: str) -> str:
            s = (prefix or "").strip() + "\u0000" + separator + "\u0000" + (target or "").strip()
            return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

        top_k = int(config.top_k)
        if top_k <= 0 or top_k >= n:
            edges: list[tuple[int, int]] = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            k = max(1, min(top_k, n - 1))
            edges_set: dict[tuple[int, int], None] = {}
            for i in range(n):
                row = cheap[i].copy()
                row[i] = -np.inf
                # argpartition gives an unordered top-k; sort them after.
                idx = np.argpartition(-row, k)[:k]
                idx = idx[np.argsort(-row[idx])]
                for j in idx:
                    if i == int(j):
                        continue
                    edges_set[(i, int(j))] = None
            edges = list(edges_set.keys())

        prefixes = [tail_text[local_idx[i]] for i, _ in edges]
        targets = [head_text[local_idx[j]] for _, j in edges]
        keys = [edge_key(p, config.lm_separator, t) for p, t in zip(prefixes, targets, strict=True)]

        # For multiple LMs, average their scores.
        total = np.zeros(len(edges), dtype=np.float32)
        for scorer in scorers:
            model_name = scorer.cfg.model_name
            kind = f"lm_ml{int(config.lm_max_length)}"
            cached = cache.get_many(kind=kind, model=model_name, keys=keys)

            missing_idx = [k for k, key in enumerate(keys) if key not in cached]
            if missing_idx:
                miss_pref = [prefixes[k] for k in missing_idx]
                miss_tgt = [targets[k] for k in missing_idx]
                scores = scorer.score_pairs(
                    miss_pref,
                    miss_tgt,
                    batch_size=config.lm_batch_size,
                    separator=config.lm_separator,
                )
                cache.set_many(
                    kind=kind,
                    model=model_name,
                    entries=[(keys[k], float(scores[i])) for i, k in enumerate(missing_idx)],
                )
                for i, k in enumerate(missing_idx):
                    cached[keys[k]] = float(scores[i])

            total += np.array([cached[k] for k in keys], dtype=np.float32)

        avg = total / float(len(scorers))
        for (i, j), s in zip(edges, avg, strict=True):
            lm_scores[i, j] = float(s)

    # Combine into weights.
    w = (cheap + (config.w_lm * lm_scores if scorers else 0.0)).astype(np.float32)

    if anchor_page_id is None:
        # Choose a plausible start node: strong overall outgoing coherence.
        start_local = int(np.argmax(w.sum(axis=1)))

    # Solve.
    if config.solve_method == "greedy":
        return _greedy_path(local_ids, w, start=start_local)
    if config.solve_method == "beam":
        return _beam_path(local_ids, w, start=start_local, beam_width=config.beam_width)

    # Default: OR-Tools.
    try:
        return _ortools_path(local_ids, w, start=start_local, time_limit_sec=config.ortools_time_limit_sec)
    except Exception:
        return _beam_path(local_ids, w, start=start_local, beam_width=config.beam_width)


def _greedy_path(node_ids: list[int], w: np.ndarray, *, start: int) -> list[int]:
    n = len(node_ids)
    used = [False] * n
    path = [start]
    used[start] = True
    for _ in range(n - 1):
        i = path[-1]
        candidates = [(w[i, j], j) for j in range(n) if not used[j] and j != i]
        if not candidates:
            break
        _, j = max(candidates)
        used[j] = True
        path.append(j)
    return [node_ids[i] for i in path]


def _beam_path(node_ids: list[int], w: np.ndarray, *, start: int, beam_width: int) -> list[int]:
    n = len(node_ids)
    beam_width = max(1, int(beam_width))

    # Each state: (score, path_indices, used_mask)
    states: list[tuple[float, list[int], int]] = [(0.0, [start], 1 << start)]

    for _ in range(n - 1):
        new_states: list[tuple[float, list[int], int]] = []
        for score, path, mask in states:
            i = path[-1]
            for j in range(n):
                if mask & (1 << j):
                    continue
                if i == j:
                    continue
                new_states.append((score + float(w[i, j]), [*path, j], mask | (1 << j)))

        if not new_states:
            break

        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]

    best = max(states, key=lambda x: x[0])
    return [node_ids[i] for i in best[1]]


def _ortools_path(node_ids: list[int], w: np.ndarray, *, start: int, time_limit_sec: int) -> list[int]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(node_ids)

    # Add a dummy end node.
    end = n
    n_total = n + 1

    # Convert weights to integer costs.
    # Lower cost is better.
    w_max = float(np.max(w[np.isfinite(w)]))
    w_min = float(np.min(w[np.isfinite(w)]))
    if not np.isfinite(w_max) or not np.isfinite(w_min):
        w_max, w_min = 0.0, -1.0

    scale = 10000.0
    big_m = int(1e9)

    def cost(i: int, j: int) -> int:
        # i,j are node indices in [0..n_total-1]
        if i == end:
            return big_m
        if j == end:
            return 0
        if i == j:
            return big_m
        ww = float(w[i, j])
        if not np.isfinite(ww):
            return big_m
        return int(round((w_max - ww) * scale))

    manager = pywrapcp.RoutingIndexManager(n_total, 1, start, end)
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return cost(i, j)

    transit_callback_index = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = int(max(1, time_limit_sec))

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools failed to find a solution")

    # Extract route.
    index = routing.Start(0)
    route_nodes: list[int] = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != end:
            route_nodes.append(node)
        index = solution.Value(routing.NextVar(index))

    return [node_ids[i] for i in route_nodes]


def write_submission(pred_order: list[int], out_csv: str | Path) -> None:
    import pandas as pd

    out_csv = Path(out_csv)
    df = pd.DataFrame({"original_page": list(range(1, len(pred_order) + 1)), "shuffled_page": pred_order})
    df.to_csv(out_csv, index=False)
