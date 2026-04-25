from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _vectorize(text: str) -> Counter[str]:
    return Counter(text.lower().split())


def cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    common = set(a.keys()).intersection(b.keys())
    num = sum(a[x] * b[x] for x in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


class DenseFallbackEncoder:
    def __init__(self, docs: list[str]) -> None:
        self.doc_vecs = [_vectorize(d) for d in docs]

    def score(self, query: str, idx: int) -> float:
        qv = _vectorize(query)
        return cosine_sim(qv, self.doc_vecs[idx])


class DenseTransformerEncoder:
    def __init__(self, docs: list[str], model_name: str) -> None:
        self._model: Any | None = None
        self._doc_embs: Any | None = None
        self._fallback = DenseFallbackEncoder(docs)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._doc_embs = self._model.encode(docs, normalize_embeddings=True)
        except Exception:
            self._model = None
            self._doc_embs = None

    def score(self, query: str, idx: int) -> float:
        if self._model is None or self._doc_embs is None:
            return self._fallback.score(query, idx)
        q = self._model.encode([query], normalize_embeddings=True)[0]
        d = self._doc_embs[idx]
        # numpy-like dot compatibility without importing numpy directly
        return float((q * d).sum())
