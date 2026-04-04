from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LMConfig:
    model_name: str
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    dtype: str = "bf16"  # bf16|fp16|fp32
    device: str | None = None
    max_length: int = 512


class LMScorer:
    """Directional boundary scorer using a causal LM.

    Scores log P(target | prefix) averaged per target token.
    """

    def __init__(self, cfg: LMConfig):
        self.cfg = cfg

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

        kwargs: dict = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if cfg.load_in_4bit:
            kwargs["load_in_4bit"] = True
        if cfg.load_in_8bit:
            kwargs["load_in_8bit"] = True

        if cfg.device is None:
            kwargs["device_map"] = "auto"
        elif cfg.device == "cpu":
            kwargs["device_map"] = {"": "cpu"}
        else:
            kwargs["device_map"] = {"": cfg.device}

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        self.model.eval()

        self._torch = torch

    def score_pairs(
        self,
        prefixes: list[str],
        targets: list[str],
        *,
        batch_size: int = 8,
        separator: str = "\n\n",
    ) -> list[float]:
        if len(prefixes) != len(targets):
            raise ValueError("prefixes and targets must have same length")

        out: list[float] = []
        for i in range(0, len(prefixes), batch_size):
            p = prefixes[i : i + batch_size]
            t = targets[i : i + batch_size]
            out.extend(self._score_batch(p, t, separator=separator))
        return out

    def _score_batch(self, prefixes: list[str], targets: list[str], *, separator: str) -> list[float]:
        torch = self._torch
        import torch.nn.functional as F

        tok = self.tokenizer

        bos = tok.bos_token_id

        batch_input_ids: list[list[int]] = []
        prefix_lens: list[int] = []
        target_lens: list[int] = []

        for pref, tgt in zip(prefixes, targets, strict=True):
            pref = (pref or "").strip()
            tgt = (tgt or "").strip()

            # Ensure a non-empty prefix so the first target token has a context.
            pref_text = (pref + separator) if pref else separator

            pref_ids = tok(pref_text, add_special_tokens=False).input_ids
            tgt_ids = tok(tgt, add_special_tokens=False).input_ids

            if bos is not None:
                pref_ids = [bos] + pref_ids

            # Truncate: keep the end of prefix and start of target.
            max_len = int(self.cfg.max_length)
            if max_len > 0 and (len(pref_ids) + len(tgt_ids) > max_len):
                overflow = len(pref_ids) + len(tgt_ids) - max_len
                # First truncate prefix from the left.
                if overflow > 0 and len(pref_ids) > 1:
                    cut = min(overflow, len(pref_ids) - 1)
                    pref_ids = pref_ids[cut:]
                    overflow -= cut
                # Then truncate target from the right.
                if overflow > 0 and len(tgt_ids) > 0:
                    tgt_ids = tgt_ids[: max(0, len(tgt_ids) - overflow)]

            if not pref_ids:
                pref_ids = [bos] if bos is not None else [tok.eos_token_id]

            batch_input_ids.append(pref_ids + tgt_ids)
            prefix_lens.append(len(pref_ids))
            target_lens.append(len(tgt_ids))

        max_len = max(len(x) for x in batch_input_ids)
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full((len(batch_input_ids), max_len), fill_value=int(pad_id), dtype=torch.long)
        attn = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)

        for i, ids in enumerate(batch_input_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1

        device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Token-level negative log-likelihood.
            vocab = shift_logits.shape[-1]
            losses = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            scores: list[float] = []
            for b in range(len(batch_input_ids)):
                tgt_len = target_lens[b]
                if tgt_len <= 0:
                    scores.append(float("-inf"))
                    continue

                pref_len = prefix_lens[b]
                start = max(0, pref_len - 1)
                end = min(losses.shape[1], pref_len + tgt_len - 1)

                l = losses[b, start:end]
                # Average log-prob per target token.
                scores.append(float((-l.sum() / max(1, (end - start))).item()))

        return scores
