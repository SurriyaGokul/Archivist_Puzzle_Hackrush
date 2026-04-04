from __future__ import annotations

import argparse
from pathlib import Path

from archivist.config import SolverConfig, load_config, save_config
from archivist.data import load_pages_csv
from archivist.solver import solve_pages, write_submission


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve a shuffled book CSV and write a submission CSV.")
    p.add_argument("--test_csv", required=True, help="Path to BookA_test.csv or BookB_test.csv")
    p.add_argument("--out_csv", required=True, help="Output path (e.g. BookA.csv)")
    p.add_argument("--config", help="Optional JSON config path")
    p.add_argument("--save_config", help="Optional path to write the effective config JSON")
    p.add_argument("--cache_dir", default=".cache", help="Cache dir for LM edge scores")

    p.add_argument("--embed_model", help="SentenceTransformer model id, or 'tfidf' for offline")
    p.add_argument("--assign_method", choices=["spectral", "nearest_anchor"], help="Chapter bucketing")
    p.add_argument("--solve_method", choices=["ortools", "beam", "greedy"], help="Path solver")

    p.add_argument("--top_k", type=int, help="LM edge pruning: score LM only for top-k cheap edges per node")

    p.add_argument(
        "--lm_models",
        default="",
        help="Comma-separated causal LM model ids for boundary scoring (optional)",
    )
    p.add_argument("--lm_batch_size", type=int, help="LM batch size")
    p.add_argument("--lm_max_length", type=int, help="LM max sequence length")
    p.add_argument("--lm_4bit", action="store_true", help="Load LM in 4-bit (if supported)")
    p.add_argument("--lm_8bit", action="store_true", help="Load LM in 8-bit (if supported)")
    p.add_argument("--lm_device", help="Force LM device (e.g. 'cuda' or 'cpu')")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SolverConfig()

    if args.embed_model:
        cfg.embed_model = args.embed_model
    if args.assign_method:
        cfg.assign_method = args.assign_method
    if args.solve_method:
        cfg.solve_method = args.solve_method
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)

    if args.lm_models:
        cfg.lm_models = [m.strip() for m in args.lm_models.split(",") if m.strip()]
    if args.lm_batch_size is not None:
        cfg.lm_batch_size = int(args.lm_batch_size)
    if args.lm_max_length is not None:
        cfg.lm_max_length = int(args.lm_max_length)
    if args.lm_4bit:
        cfg.lm_load_in_4bit = True
    if args.lm_8bit:
        cfg.lm_load_in_8bit = True
    if args.lm_device:
        cfg.lm_device = args.lm_device

    if args.save_config:
        save_config(cfg, args.save_config)

    pages = load_pages_csv(args.test_csv)
    pred = solve_pages(pages, config=cfg, cache_dir=args.cache_dir)
    write_submission(pred, Path(args.out_csv))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
