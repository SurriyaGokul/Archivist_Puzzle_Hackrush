from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

from archivist.config import SolverConfig, load_config, save_config
from archivist.data import load_pages_csv
from archivist.metrics import EvalResult, kendall_tau_normalized_score
from archivist.solver import solve_pages


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight local eval by shuffling the training book.")
    p.add_argument("--train_csv", required=True, help="Path to training CSV (ordered)")
    p.add_argument("--config", help="Optional JSON config path")
    p.add_argument("--embed_model", help="SentenceTransformer model id, or 'tfidf' for offline")
    p.add_argument("--assign_method", choices=["spectral", "nearest_anchor"], help="Chapter bucketing")
    p.add_argument("--solve_method", choices=["ortools", "beam", "greedy"], help="Path solver")
    p.add_argument(
        "--lm_models",
        default="",
        help="Comma-separated causal LM model ids for boundary scoring (optional)",
    )
    p.add_argument("--lm_4bit", action="store_true", help="Load LM in 4-bit (if supported)")
    p.add_argument("--lm_8bit", action="store_true", help="Load LM in 8-bit (if supported)")
    p.add_argument("--lm_device", help="Force LM device (e.g. 'cuda' or 'cpu')")
    p.add_argument("--runs", type=int, default=3, help="Number of random shuffles")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", help="Optional path to write eval summary")
    return p.parse_args()


def run_eval(train_csv: str | Path, *, config: SolverConfig, runs: int, seed: int) -> list[EvalResult]:
    pages_ordered = load_pages_csv(train_csv)
    true_order = [p.page_id for p in pages_ordered]

    rng = random.Random(seed)
    results: list[EvalResult] = []

    for r in range(int(runs)):
        pages = pages_ordered[:]
        rng.shuffle(pages)
        pred = solve_pages(pages, config=config)
        score = kendall_tau_normalized_score(pred, true_order)
        results.append(EvalResult(score=score, n_pages=len(pages)))

    return results


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

    if args.lm_models:
        cfg.lm_models = [m.strip() for m in args.lm_models.split(",") if m.strip()]
    if args.lm_4bit:
        cfg.lm_load_in_4bit = True
    if args.lm_8bit:
        cfg.lm_load_in_8bit = True
    if args.lm_device:
        cfg.lm_device = args.lm_device

    results = run_eval(args.train_csv, config=cfg, runs=args.runs, seed=args.seed)
    avg = sum(r.score for r in results) / max(1, len(results))

    payload = {
        "avg_score": avg,
        "runs": [asdict(r) for r in results],
        "config": asdict(cfg),
    }

    if args.out_json:
        Path(args.out_json).write_text(__import__("json").dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"avg_score={avg:.6f}")


if __name__ == "__main__":
    main()
