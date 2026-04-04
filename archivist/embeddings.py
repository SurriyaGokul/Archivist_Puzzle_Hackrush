from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


@dataclass
class Embedder:
    model_name: str
    device: str | None = None
    batch_size: int = 32

    _st_model: object | None = None
    _tfidf: object | None = None
    _warned_fallback: bool = False
    _warned_transformers: bool = False
    _hf_tok: object | None = None
    _hf_model: object | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings."""

        # Explicit offline mode.
        if self.model_name.strip().lower() == "tfidf":
            return self._encode_tfidf(texts, fit_if_needed=True)

        # Preferred: SentenceTransformers.
        try:
            from sentence_transformers import SentenceTransformer

            if self._st_model is None:
                self._st_model = SentenceTransformer(self.model_name, device=self.device)
            model = self._st_model
            emb = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) >= 64,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            # If sentence-transformers isn't available, try a Transformers-based
            # embedding path for common embedding models (e.g. BGE).
            try:
                return self._encode_transformers(texts)
            except Exception:
                pass

            # Fallback: TF-IDF (CPU, no downloads). We *cache the vectorizer* so
            # multiple encode() calls share a compatible feature space.
            if not self._warned_fallback and self.model_name.strip().lower() != "tfidf":
                self._warned_fallback = True
                import sys

                print(
                    f"[archivist] WARNING: failed to load SentenceTransformer '{self.model_name}'. "
                    f"Falling back to TF-IDF embeddings. Error: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            return self._encode_tfidf(texts, fit_if_needed=True)

    def _encode_transformers(self, texts: list[str]) -> np.ndarray:
        """Embed texts using plain Transformers (no sentence-transformers).

        This is a lightweight mean-pooling implementation that works well for
        popular embedding checkpoints like BAAI/bge-large-en-v1.5.
        """

        import torch
        from transformers import AutoModel, AutoTokenizer

        # Auto device: use GPU if available unless explicitly overridden.
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = str(device)

        if self._hf_tok is None or self._hf_model is None:
            self._hf_tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.eval()
            self._hf_model.to(device)

            if not self._warned_transformers:
                self._warned_transformers = True
                import sys

                print(
                    f"[archivist] Using Transformers embedder for '{self.model_name}' on device '{device}'",
                    file=sys.stderr,
                )

            # Speed/memory: fp16 on CUDA is usually fine for embedding models.
            if device != "cpu":
                try:
                    self._hf_model.half()
                except Exception:
                    pass

        tok = self._hf_tok
        model = self._hf_model

        batch_size = int(max(1, self.batch_size))
        out: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i : i + batch_size]]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                last = outputs.last_hidden_state  # (B, T, H)

                # Pooling: BGE models use CLS pooling; default to mean pooling.
                name = self.model_name.lower()
                use_cls = "bge" in name

                mask = enc.get("attention_mask")
                if use_cls or mask is None:
                    emb = last[:, 0, :]
                else:
                    mask_f = mask.unsqueeze(-1).to(last.dtype)
                    summed = (last * mask_f).sum(dim=1)
                    denom = mask_f.sum(dim=1).clamp(min=1.0)
                    emb = summed / denom

            emb = emb.detach().cpu().to(torch.float32).numpy()
            out.append(emb)

        emb_all = np.vstack(out).astype(np.float32)
        return _l2_normalize(emb_all)

    def _encode_tfidf(self, texts: list[str], *, fit_if_needed: bool) -> np.ndarray:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if self._tfidf is None:
                if not fit_if_needed:
                    raise RuntimeError("TF-IDF vectorizer not initialized")
                self._tfidf = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=20000,
                )
                mat = self._tfidf.fit_transform(texts)
            else:
                mat = self._tfidf.transform(texts)

            emb = mat.toarray().astype(np.float32)
            return _l2_normalize(emb)
        except Exception:
            # Last-resort fallback: simple hashing bag-of-words.
            return self._encode_hashbow(texts, dim=4096)

    def _encode_hashbow(self, texts: list[str], *, dim: int) -> np.ndarray:
        import zlib

        dim = int(dim)
        x = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                h = zlib.adler32(w.encode("utf-8")) % dim
                x[i, h] += 1.0
        return _l2_normalize(x)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Assumes a and b are L2-normalized; returns cosine similarities."""

    return a @ b.T


def spectral_seriation(
    embeddings: np.ndarray,
    *,
    knn: int = 12,
) -> tuple[list[int], np.ndarray]:
    """Spectral seriation using the Fiedler vector on a kNN graph.

    Returns:
      - order: indices sorted by coordinate
      - coord: 1D coordinate per node
    """

    n = embeddings.shape[0]
    if n <= 2:
        coord = np.arange(n, dtype=np.float32)
        return list(range(n)), coord

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, 0.0)

    knn = max(2, min(int(knn), n - 1))

    # Build symmetric kNN graph with nonnegative weights.
    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argpartition(-sim[i], knn)[:knn]
        for j in idx:
            s = sim[i, j]
            if s <= 0:
                continue
            w[i, j] = max(w[i, j], s)
            w[j, i] = max(w[j, i], s)

    d = np.diag(w.sum(axis=1))
    l = d - w

    # Full eigen-decomposition is fine for n<=300.
    vals, vecs = np.linalg.eigh(l)
    order_vals = np.argsort(vals)

    # Fiedler vector is the 2nd smallest eigenvector.
    fiedler = vecs[:, order_vals[1]].astype(np.float32)

    # Stabilize sign (deterministic).
    if float(fiedler.sum()) < 0:
        fiedler = -fiedler

    order = list(np.argsort(fiedler))
    return order, fiedler


def quantile_bins(x: np.ndarray, q: int) -> np.ndarray:
    """Return bin edges for q equal-frequency bins."""

    q = max(1, int(q))
    if q == 1:
        return np.array([-math.inf, math.inf], dtype=np.float32)
    qs = np.linspace(0, 1, q + 1)
    edges = np.quantile(x, qs)
    edges[0] = -math.inf
    edges[-1] = math.inf
    return edges.astype(np.float32)
