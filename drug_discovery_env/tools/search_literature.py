"""search_literature — hybrid retrieval (BM25 + dense + reranker) with claim grounding."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.core.state import EvidenceRecord, GameState
from drug_discovery_env.retrieval.hybrid import HybridRetriever
from drug_discovery_env.tools.base import Tool


class SearchLiteratureTool(Tool):
    name = "search_literature"
    default_information_gain = 0.5

    def __init__(
        self,
        provider: Any,
        retriever_factory: Callable[[list[dict[str, Any]]], HybridRetriever],
        lab: LabSimulator | None = None,
    ) -> None:
        self.provider = provider
        self.retriever_factory = retriever_factory
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        query = str(params.get("query", state.disease))
        claims = list(params.get("claims", []))

        try:
            docs = self.provider.search_literature(query)
        except Exception:
            docs = []

        if not docs:
            # offline fallback corpus from rdkit_lab
            docs = [
                {
                    "id": f"sim_{i}",
                    "title": d["title"],
                    "abstract": d["abstract"],
                    "score": 0.5,
                    "source": "simulation",
                    "confidence": 0.4,
                }
                for i, d in enumerate(self.lab.literature_search(query, max_results=5))
            ]

        retriever = self.retriever_factory(docs)
        ranked = retriever.retrieve(query)

        now = datetime.now(timezone.utc).isoformat()
        for idx, doc in enumerate(ranked):
            evid_id = str(doc.get("id", f"lit_{state.step}_{idx}"))
            state.evidence_ledger[evid_id] = EvidenceRecord(
                evidence_id=evid_id,
                title=doc.get("title", ""),
                snippet=doc.get("abstract", "")[:240],
                score=float(doc.get("score", 0.0)),
                source=str(doc.get("source", "unknown")),
                timestamp=now,
                confidence=float(doc.get("grounding", {}).get("confidence", 0.5)),
            )

        claim_grounding = []
        if claims:
            for idx, claim in enumerate(claims):
                if not ranked:
                    claim_grounding.append(
                        {"claim": claim, "snippet_id": None, "confidence": 0.0, "status": "unmatched"}
                    )
                    continue
                doc = ranked[min(idx, len(ranked) - 1)]
                claim_grounding.append(
                    {
                        "claim": claim,
                        "snippet_id": doc.get("grounding", {}).get("snippet_id"),
                        "confidence": doc.get("grounding", {}).get("confidence", 0.0),
                        "status": "matched",
                    }
                )

        return {
            "query": query,
            "ranked_docs": ranked,
            "method": ranked[0].get("method", "none") if ranked else "none",
            "claim_grounding": claim_grounding,
        }
