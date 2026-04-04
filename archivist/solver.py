from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from archivist.anchors import extract_anchors
from archivist.cache import SQLiteScoreCache
from archivist.config import SolverConfig
from archivist.data import normalize_text, word_window
from archivist.embeddings import Embedder, cosine_sim_matrix, spectral_seriation
from archivist.entities import auto_detect_characters, character_flow_matrix
from archivist.heuristics import boundary_overlap_matrix
from archivist.lm import LMConfig, LMScorer
from archivist.rerank import RerankConfig, Reranker
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

    # Named entity detection.
    characters = auto_detect_characters(pages) if config.w_entity > 0 else []

    # Bucketing.
    if anchors_sorted:
        next_anchor_by_chapter: dict[int, int | None] = {}
        for i, a in enumerate(anchors_sorted):
            next_anchor_by_chapter[a.chapter_num] = anchors_sorted[i + 1].page_id if i + 1 < len(anchors_sorted) else None
        first_anchor_page_id = anchors_sorted[0].page_id

        buckets = _assign_to_chapters_spectral(
            pages,
            anchors_sorted,
            full_emb,
            method=config.assign_method,
            knn=config.spectral_knn,
            dp_penalty=config.assign_dp_penalty,
        )
        if config.assign_balance:
            buckets = _balance_buckets(
                buckets,
                pages,
                full_emb,
                anchors_sorted,
                enforce_min_size=bool(getattr(config, "assign_balance_min_size", False)),
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

    # Cross-encoder rerankers.
    rerankers: list[Reranker] = []
    if config.rerank_models:
        for m in config.rerank_models:
            rcfg = RerankConfig(
                model_name=m,
                device=config.rerank_device,
                max_length=config.rerank_max_length,
            )
            rerankers.append(Reranker(rcfg))

    cache = SQLiteScoreCache(Path(cache_dir) / "scores.sqlite")

    # Solve buckets in order and concatenate.
    out_order: list[int] = []
    for key in ordered_chapter_keys:
        page_ids = buckets.get(key, [])
        if not page_ids:
            continue

        if isinstance(key, int):
            anchor_page_id = next((a.page_id for a in anchors_sorted if a.chapter_num == key), None)
            end_target_page_id = next_anchor_by_chapter.get(key)
        else:
            anchor_page_id = None
            end_target_page_id = first_anchor_page_id if key == "__pre__" and anchors_sorted else None

        seq = _order_bucket(
            page_ids,
            anchor_page_id=anchor_page_id,
            end_target_page_id=end_target_page_id,
            id_to_idx=id_to_idx,
            head_text=head_text,
            tail_text=tail_text,
            head_emb=head_emb,
            tail_emb=tail_emb,
            full_text=full_text,
            characters=characters,
            config=config,
            scorers=scorers,
            rerankers=rerankers,
            cache=cache,
        )
        out_order.extend(seq)

    cache.close()

    # Final sanity: permutation of input page ids.
    if set(out_order) != {p.page_id for p in pages} or len(out_order) != len(pages):
        raise RuntimeError("Solver produced invalid permutation")

    return out_order


def _balance_buckets(
    buckets: dict[int | str, list[int]],
    pages: list[Page],
    full_emb: np.ndarray,
    anchors: list[Anchor],
    *,
    enforce_min_size: bool = False,
) -> dict[int | str, list[int]]:
    """Optionally rebalance chapter buckets to avoid extreme bucket sizes.

    Default behavior is intentionally conservative: only trim *oversized*
    buckets. Enforcing a minimum size is optional and can be risky (it can move
    pages into the wrong chapter), so it is gated behind `enforce_min_size`.
    """

    anchor_ids = {a.page_id for a in anchors}
    chapter_keys = [k for k in buckets if isinstance(k, int)]
    if len(chapter_keys) < 2:
        return buckets

    id_to_pos = {p.page_id: i for i, p in enumerate(pages)}

    # Compute expected size (excluding anchors from the count).
    total_non_anchor = sum(
        len([pid for pid in buckets.get(k, []) if pid not in anchor_ids])
        for k in list(buckets.keys())
    )
    expected = total_non_anchor / max(1, len(chapter_keys))
    min_size = max(2, int(expected * 0.3))
    max_size = int(expected * 2.5)

    # Optional: fill only *tiny* chapters by borrowing from immediate neighbors.
    # This is disabled by default because it can hurt test performance.
    if enforce_min_size:
        chapter_keys_sorted = sorted(chapter_keys)
        ck_to_i = {ck: i for i, ck in enumerate(chapter_keys_sorted)}

        def neighbors(ck: int) -> list[int]:
            i = ck_to_i[ck]
            out: list[int] = []
            if i - 1 >= 0:
                out.append(chapter_keys_sorted[i - 1])
            if i + 1 < len(chapter_keys_sorted):
                out.append(chapter_keys_sorted[i + 1])
            return out

        for _ in range(3):
            changed = False
            for ck in chapter_keys_sorted:
                while len(buckets.get(ck, [])) < min_size:
                    # Only borrow from immediate neighbors that can spare pages.
                    donors = [d for d in neighbors(ck) if len(buckets.get(d, [])) > min_size]
                    if not donors:
                        break

                    # Target centroid.
                    tgt_pids = buckets.get(ck, [])
                    tgt_pos = [id_to_pos[pid] for pid in tgt_pids]
                    tgt_centroid = full_emb[tgt_pos].mean(axis=0, keepdims=True)
                    tgt_centroid = tgt_centroid / max(1e-12, float(np.linalg.norm(tgt_centroid)))

                    best: tuple[float, int, int] | None = None  # (gain, donor, pid)
                    for d in donors:
                        d_pids = buckets.get(d, [])
                        d_pos = [id_to_pos[pid] for pid in d_pids]
                        d_centroid = full_emb[d_pos].mean(axis=0, keepdims=True)
                        d_centroid = d_centroid / max(1e-12, float(np.linalg.norm(d_centroid)))

                        for pid in d_pids:
                            if pid in anchor_ids:
                                continue
                            pos = id_to_pos[pid]
                            v = full_emb[pos : pos + 1]
                            s_t = float((v @ tgt_centroid.T).item())
                            s_d = float((v @ d_centroid.T).item())
                            gain = s_t - s_d
                            if best is None or gain > best[0]:
                                best = (gain, d, pid)

                    if best is None:
                        break

                    _, d, pid = best
                    if len(buckets.get(d, [])) <= min_size:
                        break
                    buckets[d].remove(pid)
                    buckets[ck].append(pid)
                    changed = True

            if not changed:
                break

    # Always: trim oversized chapters (original behavior).
    for _ in range(10):
        changed = False
        for ck in chapter_keys:
            pids = buckets.get(ck, [])
            if len(pids) <= max_size:
                continue

            # Compute centroid of this chapter.
            positions = [id_to_pos[pid] for pid in pids]
            centroid = full_emb[positions].mean(axis=0, keepdims=True)
            centroid = centroid / max(1e-12, float(np.linalg.norm(centroid)))

            # Similarity of each page to centroid.
            sims = (full_emb[positions] @ centroid.T).flatten()

            # Sort non-anchor pages by similarity (ascending = furthest first).
            candidates = [
                (float(sims[i]), pids[i])
                for i in range(len(pids))
                if pids[i] not in anchor_ids
            ]
            candidates.sort()

            overflow = len(pids) - max_size
            for _, pid in candidates[:overflow]:
                pos = id_to_pos[pid]
                page_emb = full_emb[pos : pos + 1]

                best_key = None
                best_sim = -np.inf
                for other_ck in chapter_keys:
                    if other_ck == ck:
                        continue
                    if len(buckets.get(other_ck, [])) >= max_size:
                        continue
                    other_positions = [id_to_pos[p] for p in buckets.get(other_ck, [])]
                    if not other_positions:
                        continue
                    other_centroid = full_emb[other_positions].mean(axis=0, keepdims=True)
                    other_centroid = other_centroid / max(1e-12, float(np.linalg.norm(other_centroid)))
                    s = float((page_emb @ other_centroid.T).item())
                    if s > best_sim:
                        best_sim = s
                        best_key = other_ck

                if best_key is not None:
                    buckets[ck].remove(pid)
                    buckets[best_key].append(pid)
                    changed = True

        if not changed:
            break

    return buckets


def _assign_to_chapters_spectral(
    pages: list[Page],
    anchors: list[Anchor],
    full_emb: np.ndarray,
    *,
    method: str,
    knn: int,
    dp_penalty: float,
) -> dict[int | str, list[int]]:
    """Assign each page to a chapter bucket.

    Returns buckets keyed by chapter_num plus '__pre__'/'__post__'.
    """

    buckets: dict[int | str, list[int]] = {"__pre__": [], "__post__": []}
    for a in anchors:
        buckets[a.chapter_num] = []

    if method not in {"spectral", "spectral_dp", "nearest_anchor"}:
        method = "spectral"

    if method == "nearest_anchor":
        # Nearest-anchor assignment in cosine embedding space.
        #
        # Important: chapter-heading pages share boilerplate ("Chapter X"), which
        # can make raw anchor embeddings overly similar. To make the assignment
        # more robust, we do a tiny EM-like refinement:
        #   1) assign pages to the closest anchor prototype
        #   2) recompute a per-chapter prototype from assigned pages, while
        #      keeping the original anchor embedding as an "anchor" prior.
        page_ids = [p.page_id for p in pages]
        id_to_pos = {pid: i for i, pid in enumerate(page_ids)}

        anchors_sorted = sorted(anchors, key=lambda a: a.chapter_num)
        anchor_ids = [a.page_id for a in anchors_sorted]
        anchor_pos = [id_to_pos[pid] for pid in anchor_ids]

        anchor_emb = full_emb[anchor_pos].astype(np.float32)
        proto = anchor_emb.copy()  # (K, D)

        fixed_label: dict[int, int] = {a.page_id: idx for idx, a in enumerate(anchors_sorted)}
        k_ch = len(anchors_sorted)

        # Two lightweight refinement iterations are enough for this dataset size.
        n_iter = 2
        mix = 0.7  # weight of the data-driven centroid vs the anchor prior

        assign = np.zeros(len(page_ids), dtype=np.int16)
        for _ in range(n_iter):
            sim = (full_emb @ proto.T).astype(np.float32)
            for i, pid in enumerate(page_ids):
                if pid in fixed_label:
                    assign[i] = int(fixed_label[pid])
                else:
                    assign[i] = int(np.argmax(sim[i]))

            new_proto = proto.copy()
            for r in range(k_ch):
                idxs = np.where(assign == r)[0]
                if idxs.size == 0:
                    new_proto[r] = anchor_emb[r]
                    continue
                mean = full_emb[idxs].mean(axis=0).astype(np.float32)
                denom = float(np.linalg.norm(mean))
                if denom > 1e-12:
                    mean = mean / denom
                v = ((1.0 - mix) * anchor_emb[r] + mix * mean).astype(np.float32)
                denom2 = float(np.linalg.norm(v))
                if denom2 > 1e-12:
                    v = v / denom2
                new_proto[r] = v

            proto = new_proto

        # Final assignment.
        sim = (full_emb @ proto.T).astype(np.float32)
        for i, pid in enumerate(page_ids):
            if pid in fixed_label:
                chap = anchors_sorted[int(fixed_label[pid])].chapter_num
            else:
                chap = anchors_sorted[int(np.argmax(sim[i]))].chapter_num
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

    if method == "spectral_dp":
        # The DP below assumes that chapter anchors appear in the same order as
        # chapter numbers along the spectral coordinate. In practice, chapter-
        # heading pages are stylistically similar and can cluster together,
        # producing many anchor-coordinate inversions. When that happens, the
        # monotone DP becomes infeasible and collapses assignments (e.g. most
        # pages end up in '__pre__' or the last chapter).
        #
        # Guardrail: if anchors are not monotone along the coordinate, fall
        # back to the robust 'nearest_anchor' bucketing.
        if len(anchor_coords) >= 2:
            inv = int(np.sum(anchor_coords[1:] < anchor_coords[:-1]))
            if inv > 0:
                return _assign_to_chapters_spectral(
                    pages,
                    anchors,
                    full_emb,
                    method="nearest_anchor",
                    knn=knn,
                    dp_penalty=dp_penalty,
                )

    if method == "spectral_dp":
        # DP assignment along the spectral coordinate with a mild penalty for
        # deviating from the coordinate-derived chapter.
        page_ids = [p.page_id for p in pages]
        id_to_pos = {pid: i for i, pid in enumerate(page_ids)}
        anchor_ids_set = {a.page_id for a in anchors}
        anchor_pos = [id_to_pos[a.page_id] for a in anchors]

        anchor_emb = full_emb[anchor_pos]
        sim_to_anchor = (full_emb @ anchor_emb.T).astype(np.float32)  # (N,K)

        # Sort pages by coordinate.
        order = list(np.argsort(coord))

        pre_ids: list[int] = []
        dp_indices: list[int] = []
        for i in order:
            i = int(i)
            pid = pages[i].page_id
            c = float(coord[i])
            if c < anchor_fit_list[0] and pid not in anchor_ids_set:
                pre_ids.append(pid)
            else:
                dp_indices.append(i)

        k_ch = len(anchors)
        fixed_label: dict[int, int] = {a.page_id: idx for idx, a in enumerate(anchors)}
        penalty = float(dp_penalty)

        def base_label(c: float) -> int:
            b = bisect.bisect_right(anchor_fit_list, c) - 1
            return int(max(0, min(k_ch - 1, b)))

        n2 = len(dp_indices)
        dp = np.full((n2, k_ch), -np.inf, dtype=np.float32)
        back = np.full((n2, k_ch), -1, dtype=np.int16)

        for t, i in enumerate(dp_indices):
            pid = pages[i].page_id
            b = base_label(float(coord[i]))
            scores = sim_to_anchor[i].copy()
            if penalty > 0:
                for r in range(k_ch):
                    scores[r] -= penalty * float(abs(r - b))

            if pid in fixed_label:
                r_fix = fixed_label[pid]
                masked = np.full(k_ch, -np.inf, dtype=np.float32)
                masked[r_fix] = scores[r_fix]
                scores = masked

            if t == 0:
                dp[t] = scores
                continue

            prev = dp[t - 1]
            best_val = -np.inf
            best_idx = -1
            for r in range(k_ch):
                if float(prev[r]) > best_val:
                    best_val = float(prev[r])
                    best_idx = r
                if np.isfinite(scores[r]) and np.isfinite(best_val):
                    dp[t, r] = scores[r] + best_val
                    back[t, r] = best_idx

        last_r = int(np.argmax(dp[n2 - 1]))
        labels_by_index: dict[int, int] = {}
        r = last_r
        for t in range(n2 - 1, -1, -1):
            i = dp_indices[t]
            labels_by_index[i] = int(r)
            if t > 0:
                r = int(back[t, r])

        pre_set = set(pre_ids)
        buckets["__pre__"] = pre_ids
        for i, p in enumerate(pages):
            if p.page_id in pre_set:
                continue
            if p.page_id in anchor_ids_set:
                chap = next(a.chapter_num for a in anchors if a.page_id == p.page_id)
                buckets[chap].append(p.page_id)
                continue
            lab = labels_by_index.get(i)
            if lab is None:
                lab = base_label(float(coord[i]))
            chap = anchors[int(lab)].chapter_num
            buckets[chap].append(p.page_id)

        return buckets

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
    end_target_page_id: int | None,
    id_to_idx: dict[int, int],
    head_text: list[str],
    tail_text: list[str],
    head_emb: np.ndarray,
    tail_emb: np.ndarray,
    full_text: list[str],
    characters: list[str],
    config: SolverConfig,
    scorers: list[LMScorer],
    rerankers: list[Reranker],
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

    # Named entity (character) flow feature.
    entity_score = np.zeros((n, n), dtype=np.float32)
    if characters and config.w_entity > 0:
        entity_score = character_flow_matrix(
            texts,
            characters,
            tail_words=config.entity_window,
            head_words=config.entity_window,
        )

    cheap = (config.w_emb * sim + config.w_overlap * overlap_norm + config.w_entity * entity_score).astype(np.float32)

    # Optional end bonus: encourage the last page in this bucket to transition
    # well into the next chapter anchor (or into Chapter I for the preface).
    end_bonus: np.ndarray | None = None
    if end_target_page_id is not None and end_target_page_id in id_to_idx:
        tgt_global = id_to_idx[end_target_page_id]
        tgt_head_text = head_text[tgt_global]

        end_sim = (tail_emb[local_idx] @ head_emb[tgt_global]).astype(np.float32)

        # Exact boundary overlap from each page -> target.
        ow = max(1, int(config.overlap_words))
        mo = max(1, int(config.max_overlap))
        tgt_words = normalize_text(full_text[tgt_global]).lower().split()
        tgt_prefix = tgt_words[:ow]
        end_overlap = np.zeros(n, dtype=np.float32)
        if tgt_prefix:
            for ii, gi in enumerate(local_idx):
                s_words = normalize_text(full_text[gi]).lower().split()
                s_suffix = s_words[-ow:]
                if not s_suffix:
                    continue
                kmax = min(mo, len(s_suffix), len(tgt_prefix))
                best = 0
                for k2 in range(kmax, 0, -1):
                    if s_suffix[-k2:] == tgt_prefix[:k2]:
                        best = k2
                        break
                end_overlap[ii] = float(best)

        if config.max_overlap > 0:
            end_overlap = end_overlap / float(config.max_overlap)

        end_bonus = (config.w_emb * end_sim + config.w_overlap * end_overlap).astype(np.float32)

        if scorers:
            import hashlib

            def edge_key(prefix: str, separator: str, target: str) -> str:
                s = (prefix or "").strip() + "\u0000" + separator + "\u0000" + (target or "").strip()
                return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

            prefixes_end = [tail_text[gi] for gi in local_idx]
            targets_end = [tgt_head_text] * n
            keys_end = [edge_key(p, config.lm_separator, t) for p, t in zip(prefixes_end, targets_end, strict=True)]

            # Baseline for PMI (target unconditional).
            base_key = edge_key("", config.lm_separator, tgt_head_text)

            total_end = np.zeros(n, dtype=np.float32)
            for scorer in scorers:
                model_name = scorer.cfg.model_name
                kind = f"lm_ml{int(config.lm_max_length)}"

                cached = cache.get_many(kind=kind, model=model_name, keys=keys_end)
                missing = [k for k, kk in enumerate(keys_end) if kk not in cached]
                if missing:
                    miss_pref = [prefixes_end[k] for k in missing]
                    miss_tgt = [targets_end[k] for k in missing]
                    scores = scorer.score_pairs(
                        miss_pref,
                        miss_tgt,
                        batch_size=config.lm_batch_size,
                        separator=config.lm_separator,
                    )
                    cache.set_many(
                        kind=kind,
                        model=model_name,
                        entries=[(keys_end[k], float(scores[i])) for i, k in enumerate(missing)],
                    )
                    for i, k in enumerate(missing):
                        cached[keys_end[k]] = float(scores[i])

                cond = np.array([cached[kk] for kk in keys_end], dtype=np.float32)

                if config.lm_use_pmi:
                    base = cache.get_many(kind=kind, model=model_name, keys=[base_key])
                    if base_key not in base:
                        s0 = scorer.score_pairs(
                            [""],
                            [tgt_head_text],
                            batch_size=1,
                            separator=config.lm_separator,
                        )[0]
                        cache.set_many(kind=kind, model=model_name, entries=[(base_key, float(s0))])
                        base[base_key] = float(s0)
                    cond = cond - float(base[base_key])

                total_end += cond

            end_bonus = end_bonus + (config.w_lm * (total_end / float(len(scorers)))).astype(np.float32)

    # LM and reranker features (optional).
    # We use the same candidate edge set (controlled by `top_k`) for both.
    lm_scores = np.zeros((n, n), dtype=np.float32)
    rr_scores = np.zeros((n, n), dtype=np.float32)
    if scorers or rerankers:
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

        # LM scoring.
        if scorers:
            keys = [edge_key(p, config.lm_separator, t) for p, t in zip(prefixes, targets, strict=True)]

            base_targets = [head_text[local_idx[j]] for j in range(n)]
            base_keys = [edge_key("", config.lm_separator, t) for t in base_targets]
            tgt_idx = np.array([j for _, j in edges], dtype=np.int32)

            # For multiple LMs, average their scores.
            total = np.zeros(len(edges), dtype=np.float32)
            for scorer in scorers:
                model_name = scorer.cfg.model_name
                kind = f"lm_ml{int(config.lm_max_length)}"

                baseline: np.ndarray | None = None
                if config.lm_use_pmi:
                    cached_base = cache.get_many(kind=kind, model=model_name, keys=base_keys)
                    missing_base = [j for j, k in enumerate(base_keys) if k not in cached_base]
                    if missing_base:
                        miss_pref = [""] * len(missing_base)
                        miss_tgt = [base_targets[j] for j in missing_base]
                        scores = scorer.score_pairs(
                            miss_pref,
                            miss_tgt,
                            batch_size=config.lm_batch_size,
                            separator=config.lm_separator,
                        )
                        cache.set_many(
                            kind=kind,
                            model=model_name,
                            entries=[(base_keys[j], float(scores[i])) for i, j in enumerate(missing_base)],
                        )
                        for i, j in enumerate(missing_base):
                            cached_base[base_keys[j]] = float(scores[i])
                    baseline = np.array([cached_base[k] for k in base_keys], dtype=np.float32)

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

                cond = np.array([cached[k] for k in keys], dtype=np.float32)
                if baseline is not None:
                    cond = cond - baseline[tgt_idx]
                total += cond

            avg = total / float(len(scorers))
            for (i, j), s in zip(edges, avg, strict=True):
                lm_scores[i, j] = float(s)

        # Cross-encoder reranker scoring.
        if rerankers and float(config.w_rerank) != 0.0:
            import hashlib

            def rr_key(left: str, right: str) -> str:
                s = (left or "").strip() + "\u0000" + (right or "").strip()
                return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

            rr_keys = [rr_key(p, t) for p, t in zip(prefixes, targets, strict=True)]

            total_rr = np.zeros(len(edges), dtype=np.float32)
            for rr in rerankers:
                model_name = rr.cfg.model_name
                kind = f"rr_ml{int(rr.cfg.max_length)}"
                cached = cache.get_many(kind=kind, model=model_name, keys=rr_keys)
                missing_idx = [k for k, key in enumerate(rr_keys) if key not in cached]
                if missing_idx:
                    miss_l = [prefixes[k] for k in missing_idx]
                    miss_r = [targets[k] for k in missing_idx]
                    scores = rr.score_pairs(miss_l, miss_r, batch_size=config.rerank_batch_size)
                    cache.set_many(
                        kind=kind,
                        model=model_name,
                        entries=[(rr_keys[k], float(scores[i])) for i, k in enumerate(missing_idx)],
                    )
                    for i, k in enumerate(missing_idx):
                        cached[rr_keys[k]] = float(scores[i])

                cond = np.array([cached[k] for k in rr_keys], dtype=np.float32)
                total_rr += cond

            avg_rr = total_rr / float(len(rerankers))
            for (i, j), s in zip(edges, avg_rr, strict=True):
                rr_scores[i, j] = float(s)

    # Combine into weights.
    w = (cheap + (config.w_lm * lm_scores if scorers else 0.0) + (config.w_rerank * rr_scores)).astype(np.float32)

    if anchor_page_id is None:
        # Choose a plausible start node: strong overall outgoing coherence.
        start_local = int(np.argmax(w.sum(axis=1)))

    # Solve.
    if config.solve_method == "greedy":
        seq = _greedy_path(local_ids, w, start=start_local)
    elif config.solve_method == "beam":
        seq = _beam_path(local_ids, w, start=start_local, beam_width=config.beam_width, end_bonus=end_bonus)
    else:
        # Default: OR-Tools.
        try:
            seq = _ortools_path(
                local_ids,
                w,
                start=start_local,
                time_limit_sec=config.ortools_time_limit_sec,
                end_bonus=end_bonus,
            )
        except Exception:
            seq = _beam_path(local_ids, w, start=start_local, beam_width=config.beam_width, end_bonus=end_bonus)

    # Optional local refinement.
    if int(config.refine_window) > 0 and len(seq) >= 3:
        fixed_prefix = 1 if (anchor_page_id is not None and seq and seq[0] == anchor_page_id) else 0
        id_to_local = {pid: i for i, pid in enumerate(local_ids)}
        path_idx = [id_to_local[pid] for pid in seq]
        path_idx = _refine_sliding_window(
            path_idx,
            w,
            window=int(config.refine_window),
            passes=int(max(1, config.refine_passes)),
            fixed_prefix=fixed_prefix,
        )
        seq = [local_ids[i] for i in path_idx]

    return seq


def _refine_sliding_window(
    path: list[int],
    w: np.ndarray,
    *,
    window: int,
    passes: int,
    fixed_prefix: int,
) -> list[int]:
    n = len(path)
    window = int(window)
    if window <= 2 or n <= window:
        return path

    fixed_prefix = int(max(0, fixed_prefix))
    passes = int(max(1, passes))

    def score_path(p: list[int]) -> float:
        s = 0.0
        for a, b in zip(p, p[1:]):
            s += float(w[a, b])
        return s

    for _ in range(passes):
        improved = False
        for start in range(fixed_prefix, n - window + 1):
            win = path[start : start + window]
            prev = path[start - 1] if start > 0 else None
            nxt = path[start + window] if start + window < n else None

            old_score = 0.0
            if prev is not None:
                old_score += float(w[prev, win[0]])
            old_score += score_path(win)
            if nxt is not None:
                old_score += float(w[win[-1], nxt])

            new_win = _best_window_order(win, w, prev=prev, nxt=nxt)
            if new_win == win:
                continue

            new_score = 0.0
            if prev is not None:
                new_score += float(w[prev, new_win[0]])
            new_score += score_path(new_win)
            if nxt is not None:
                new_score += float(w[new_win[-1], nxt])

            if new_score > old_score + 1e-6:
                path[start : start + window] = new_win
                improved = True

        if not improved:
            break

    return path


def _best_window_order(win: list[int], w: np.ndarray, *, prev: int | None, nxt: int | None) -> list[int]:
    # Exact DP for best permutation of `win` given optional fixed neighbors prev/nxt.
    m = len(win)
    if m <= 2:
        return win

    size = 1 << m
    dp = np.full((size, m), -np.inf, dtype=np.float32)
    parent = np.full((size, m), -1, dtype=np.int16)

    for j in range(m):
        start_score = float(w[prev, win[j]]) if prev is not None else 0.0
        dp[1 << j, j] = start_score

    for mask in range(size):
        for last in range(m):
            cur = float(dp[mask, last])
            if not np.isfinite(cur):
                continue
            for j in range(m):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                cand = cur + float(w[win[last], win[j]])
                if cand > float(dp[nmask, j]):
                    dp[nmask, j] = cand
                    parent[nmask, j] = last

    full = size - 1
    best = -np.inf
    best_last = 0
    for last in range(m):
        cur = float(dp[full, last])
        if not np.isfinite(cur):
            continue
        if nxt is not None:
            cur += float(w[win[last], nxt])
        if cur > best:
            best = cur
            best_last = last

    # Reconstruct.
    out_idx: list[int] = []
    mask = full
    last = best_last
    while True:
        out_idx.append(last)
        p = int(parent[mask, last])
        mask = mask ^ (1 << last)
        if mask == 0:
            break
        last = p

    out_idx.reverse()
    return [win[i] for i in out_idx]


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


def _beam_path(
    node_ids: list[int],
    w: np.ndarray,
    *,
    start: int,
    beam_width: int,
    end_bonus: np.ndarray | None = None,
) -> list[int]:
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

    if end_bonus is None:
        best = max(states, key=lambda x: x[0])
    else:
        best = max(states, key=lambda x: x[0] + float(end_bonus[x[1][-1]]))
    return [node_ids[i] for i in best[1]]


def _ortools_path(
    node_ids: list[int],
    w: np.ndarray,
    *,
    start: int,
    time_limit_sec: int,
    end_bonus: np.ndarray | None = None,
) -> list[int]:
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

    if end_bonus is not None:
        eb = end_bonus[np.isfinite(end_bonus)]
        if eb.size:
            w_max = max(w_max, float(np.max(eb)))

    scale = 10000.0
    big_m = int(1e9)

    def cost(i: int, j: int) -> int:
        # i,j are node indices in [0..n_total-1]
        if i == end:
            return big_m
        if j == end:
            if end_bonus is None:
                return 0
            ww = float(end_bonus[i])
            if not np.isfinite(ww):
                return big_m
            return int(round((w_max - ww) * scale))
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
