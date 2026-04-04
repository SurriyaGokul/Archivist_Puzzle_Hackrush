from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SolverConfig:
    # Text windows (word-based to stay model-agnostic).
    head_words: int = 120
    tail_words: int = 120

    # Candidate pruning.
    top_k: int = 30

    # Feature weights for edge weight = w_lm*lm + w_emb*sim + w_overlap*overlap_norm
    w_lm: float = 1.0
    w_emb: float = 0.7
    w_overlap: float = 0.5

    # Boundary overlap heuristic.
    overlap_words: int = 80
    max_overlap: int = 12

    # Embedding model.
    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_batch_size: int = 32

    # LM scoring.
    lm_models: list[str] = field(default_factory=list)
    lm_batch_size: int = 8
    lm_load_in_4bit: bool = False
    lm_load_in_8bit: bool = False
    lm_dtype: str = "bf16"  # bf16|fp16|fp32
    lm_device: str | None = None  # e.g. 'cuda', 'cpu', or None for auto
    lm_max_length: int = 512
    lm_separator: str = "\n\n"

    # Chapter bucketing.
    assign_method: str = "spectral"  # spectral|nearest_anchor
    spectral_knn: int = 12

    # Path solving.
    solve_method: str = "ortools"  # ortools|beam|greedy
    ortools_time_limit_sec: int = 10
    beam_width: int = 10

    # Repro.
    seed: int = 42


def save_config(config: SolverConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: str | Path) -> SolverConfig:
    import json

    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return SolverConfig(**data)
