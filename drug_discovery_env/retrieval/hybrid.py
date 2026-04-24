from __future__ import annotations

from typing import Any

from drug_discovery_env.config.settings import RetrievalMode, Settings
from drug_discovery_env.retrieval.dense import DenseFallbackEncoder, DenseTransformerEncoder
from drug_discovery_env.retrieval.lexical import BM25Lite
from drug_discovery_env.retrieval.rerank import CrossEncoderReranker, fallback_rerank_score


class HybridRetriever:
    def __init__(self, settings: Settings, documents: list[dict[str, Any]]) -> None:
        self.settings = settings
        self.documents = documents
        self.texts = [f"{d.get('title', '')} {d.get('abstract', '')}" for d in documents]
        self.bm25 = BM25Lite(self.texts)
        if settings.retrieval.use_transformer_models:
            self.dense = DenseTransformerEncoder(self.texts, settings.retrieval.dense_model_name)
            self.reranker = CrossEncoderReranker(settings.retrieval.reranker_model_name)
        else:
            self.dense = DenseFallbackEncoder(self.texts)
            self.reranker = None

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        mode = self.settings.retrieval.mode
        weighted: list[tuple[float, dict[str, Any]]] = []
        for idx, doc in enumerate(self.documents):
            lexical = self.bm25.score(query, idx)
            dense = self.dense.score(query, idx)
            rerank = (
                self.reranker.score(query, self.texts[idx])
                if self.reranker is not None
                else fallback_rerank_score(query, self.texts[idx])
            )

            if mode == RetrievalMode.LEXICAL_ONLY:
                score = lexical
                method = "lexical"
            elif mode == RetrievalMode.DENSE_ONLY:
                score = dense
                method = "dense"
            else:
                score = (
                    self.settings.retrieval.bm25_weight * lexical
                    + self.settings.retrieval.dense_weight * dense
                    + self.settings.retrieval.rerank_weight * rerank
                )
                method = "hybrid"

            row = dict(doc)
            row["score"] = score
            row["method"] = method
            row["grounding"] = {
                "snippet_id": row.get("id", f"doc_{idx}"),
                "confidence": max(0.0, min(1.0, score)),
            }
            weighted.append((score, row))

        weighted.sort(key=lambda x: x[0], reverse=True)
        top_k = self.settings.retrieval.top_k
        ranked = [x[1] for x in weighted[:top_k]]
        if len(ranked) < self.settings.retrieval.fallback_min_docs and mode == RetrievalMode.HYBRID:
            fallback = sorted(weighted, key=lambda x: x[1].get("score", 0.0), reverse=True)
            ranked = [x[1] for x in fallback[: self.settings.retrieval.fallback_min_docs]]
            for item in ranked:
                item["method"] = "hybrid_fallback"
        return ranked
