from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archivist.config import SolverConfig, load_config
from archivist.data import load_pages_csv
from archivist.solver import solve_pages, write_submission


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve both BookA and BookB test sets.")
    p.add_argument("--test_dir", default="Test Data", help="Directory containing BookA_test.csv and BookB_test.csv")
    p.add_argument("--out_dir", default=".", help="Directory to write BookA.csv and BookB.csv")
    p.add_argument("--config", help="Optional JSON config path")

    p.add_argument("--head_words", type=int, help="Head window size (words)")
    p.add_argument("--tail_words", type=int, help="Tail window size (words)")

    p.add_argument("--embed_model", help="SentenceTransformer model id, or 'tfidf' for offline")
    p.add_argument("--assign_method", choices=["spectral", "spectral_dp", "nearest_anchor"], help="Chapter bucketing")
    p.add_argument("--solve_method", choices=["ortools", "beam", "greedy"], help="Path solver")
    p.add_argument("--cache_dir", default=".cache", help="Cache dir for LM edge scores")

    p.add_argument(
        "--assign_dp_penalty",
        type=float,
        help="spectral_dp only: penalty per-chapter for deviating from coordinate-derived chapter",
    )

    p.add_argument("--w_lm", type=float, help="Weight for LM feature")
    p.add_argument("--w_emb", type=float, help="Weight for embedding cosine feature")
    p.add_argument("--w_overlap", type=float, help="Weight for boundary word-overlap feature")
    p.add_argument("--overlap_words", type=int, help="Boundary overlap window (words)")
    p.add_argument("--max_overlap", type=int, help="Max exact overlap length")

    p.add_argument("--spectral_knn", type=int, help="kNN for spectral bucketing")

    p.add_argument("--top_k", type=int, help="LM edge pruning: score LM only for top-k cheap edges per node")

    p.add_argument(
        "--lm_models",
        default="",
        help="Comma-separated causal LM model ids for boundary scoring (optional)",
    )
    p.add_argument("--lm_batch_size", type=int, help="LM batch size")
    p.add_argument("--lm_max_length", type=int, help="LM max sequence length")
    p.add_argument("--lm_pmi", action="store_true", help="Use PMI: logP(tgt|prefix)-logP(tgt)")
    p.add_argument("--lm_4bit", action="store_true", help="Load LM in 4-bit (if supported)")
    p.add_argument("--lm_8bit", action="store_true", help="Load LM in 8-bit (if supported)")
    p.add_argument("--lm_device", help="Force LM device (e.g. 'cuda' or 'cpu')")

    p.add_argument(
        "--rerank_models",
        default="",
        help="Comma-separated cross-encoder reranker model ids (optional)",
    )
    p.add_argument("--rerank_batch_size", type=int, help="Reranker batch size")
    p.add_argument("--rerank_max_length", type=int, help="Reranker max sequence length")
    p.add_argument("--rerank_device", help="Force reranker device (e.g. 'cuda' or 'cpu')")
    p.add_argument("--w_rerank", type=float, help="Weight for reranker feature")

    p.add_argument("--w_entity", type=float, help="Weight for named-entity flow feature")
    p.add_argument("--entity_window", type=int, help="Words to scan at tail/head for entity detection")
    p.add_argument("--no_balance", action="store_true", help="Disable chapter balance enforcement")

    p.add_argument("--refine_window", type=int, help="Sliding-window exact refinement (0 disables)")
    p.add_argument("--refine_passes", type=int, help="Refinement passes")

    p.add_argument("--ortools_time_limit_sec", type=int, help="OR-Tools time limit per bucket")
    p.add_argument("--beam_width", type=int, help="Beam width (if using --solve_method beam)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SolverConfig()

    if args.embed_model:
        cfg.embed_model = args.embed_model
    if args.head_words is not None:
        cfg.head_words = int(args.head_words)
    if args.tail_words is not None:
        cfg.tail_words = int(args.tail_words)
    if args.assign_method:
        cfg.assign_method = args.assign_method
    if args.solve_method:
        cfg.solve_method = args.solve_method
    if args.assign_dp_penalty is not None:
        cfg.assign_dp_penalty = float(args.assign_dp_penalty)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)

    if args.w_lm is not None:
        cfg.w_lm = float(args.w_lm)
    if args.w_emb is not None:
        cfg.w_emb = float(args.w_emb)
    if args.w_overlap is not None:
        cfg.w_overlap = float(args.w_overlap)
    if args.overlap_words is not None:
        cfg.overlap_words = int(args.overlap_words)
    if args.max_overlap is not None:
        cfg.max_overlap = int(args.max_overlap)
    if args.spectral_knn is not None:
        cfg.spectral_knn = int(args.spectral_knn)

    if args.lm_models:
        cfg.lm_models = [m.strip() for m in args.lm_models.split(",") if m.strip()]
    if args.lm_batch_size is not None:
        cfg.lm_batch_size = int(args.lm_batch_size)
    if args.lm_max_length is not None:
        cfg.lm_max_length = int(args.lm_max_length)
    if args.lm_pmi:
        cfg.lm_use_pmi = True
    if args.lm_4bit:
        cfg.lm_load_in_4bit = True
    if args.lm_8bit:
        cfg.lm_load_in_8bit = True
    if args.lm_device:
        cfg.lm_device = args.lm_device

    if args.rerank_models:
        cfg.rerank_models = [m.strip() for m in args.rerank_models.split(",") if m.strip()]
    if args.rerank_batch_size is not None:
        cfg.rerank_batch_size = int(args.rerank_batch_size)
    if args.rerank_max_length is not None:
        cfg.rerank_max_length = int(args.rerank_max_length)
    if args.rerank_device:
        cfg.rerank_device = args.rerank_device
    if args.w_rerank is not None:
        cfg.w_rerank = float(args.w_rerank)

    if args.w_entity is not None:
        cfg.w_entity = float(args.w_entity)
    if args.entity_window is not None:
        cfg.entity_window = int(args.entity_window)
    if args.no_balance:
        cfg.assign_balance = False

    if args.refine_window is not None:
        cfg.refine_window = int(args.refine_window)
    if args.refine_passes is not None:
        cfg.refine_passes = int(args.refine_passes)

    if args.ortools_time_limit_sec is not None:
        cfg.ortools_time_limit_sec = int(args.ortools_time_limit_sec)
    if args.beam_width is not None:
        cfg.beam_width = int(args.beam_width)

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["BookA", "BookB"]:
        test_csv = test_dir / f"{name}_test.csv"
        out_csv = out_dir / f"{name}.csv"
        pages = load_pages_csv(test_csv)
        pred = solve_pages(pages, config=cfg, cache_dir=args.cache_dir)
        write_submission(pred, out_csv)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
