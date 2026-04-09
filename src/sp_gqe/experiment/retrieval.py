"""FAISS inner-product search on L2-normalized vectors (= cosine)."""

from __future__ import annotations

import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._texts: list[str] = []

    def add(self, vectors: np.ndarray, texts: list[str]) -> None:
        if len(texts) != len(vectors):
            raise ValueError("vectors and texts length mismatch")
        self._texts.extend(texts)
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, idx = self.index.search(q, min(k, len(self._texts)))
        out: list[tuple[str, float]] = []
        for j, sc in zip(idx[0], scores[0]):
            if j < 0:
                continue
            out.append((self._texts[j], float(sc)))
        return out
