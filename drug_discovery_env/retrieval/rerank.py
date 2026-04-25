from __future__ import annotations

from typing import Any


def fallback_rerank_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    d = set(text.lower().split())
    if not q:
        return 0.0
    precision = len(q & d) / max(1, len(d))
    recall = len(q & d) / len(q)
    return 0.5 * precision + 0.5 * recall


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model: Any | None = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def score(self, query: str, text: str) -> float:
        if self._model is None:
            return fallback_rerank_score(query, text)
        pred = self._model.predict([(query, text)])
        return float(pred[0])
