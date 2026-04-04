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

    p.add_argument("--embed_model", help="SentenceTransformer model id, or 'tfidf' for offline")
    p.add_argument("--assign_method", choices=["spectral", "nearest_anchor"], help="Chapter bucketing")
    p.add_argument("--solve_method", choices=["ortools", "beam", "greedy"], help="Path solver")
    p.add_argument("--cache_dir", default=".cache", help="Cache dir for LM edge scores")

    p.add_argument(
        "--lm_models",
        default="",
        help="Comma-separated causal LM model ids for boundary scoring (optional)",
    )
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

    if args.lm_models:
        cfg.lm_models = [m.strip() for m in args.lm_models.split(",") if m.strip()]
    if args.lm_4bit:
        cfg.lm_load_in_4bit = True
    if args.lm_8bit:
        cfg.lm_load_in_8bit = True
    if args.lm_device:
        cfg.lm_device = args.lm_device

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
