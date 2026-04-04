from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankConfig:
    model_name: str
    device: str | None = None
    max_length: int = 512


class Reranker:
    """Pairwise coherence scorer using a cross-encoder reranker.

    Returns a scalar score where larger means "more likely".

    Intended usage:
      score(tail_text(page_i), head_text(page_j))
    """

    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.model.eval()
        self.model.to(self.device)

        if self.device != "cpu":
            try:
                self.model.half()
            except Exception:
                pass

        self._torch = torch

    def score_pairs(
        self,
        left: list[str],
        right: list[str],
        *,
        batch_size: int = 16,
    ) -> list[float]:
        if len(left) != len(right):
            raise ValueError("left and right must have same length")

        torch = self._torch
        bs = int(max(1, batch_size))
        max_len = int(max(8, self.cfg.max_length))

        out: list[float] = []
        for i in range(0, len(left), bs):
            l = [str(x or "") for x in left[i : i + bs]]
            r = [str(x or "") for x in right[i : i + bs]]
            enc = self.tokenizer(
                l,
                r,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits

            if logits.ndim == 2 and logits.shape[1] > 1:
                # Treat the last logit as the "positive" score.
                score = logits[:, -1]
            else:
                score = logits.view(-1)

            out.extend([float(x) for x in score.detach().cpu().to(torch.float32).numpy().tolist()])

        return out
