from __future__ import annotations

from datetime import datetime, timezone

import pytest

from drug_discovery_env.data_provider.live_provider import LiveAPIProvider


@pytest.fixture(autouse=True)
def mock_live_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    def fake_target(self: LiveAPIProvider, disease: str) -> dict[str, object]:
        return {
            "disease": disease,
            "target": "INSR" if "diabetes" in disease.lower() else "EGFR",
            "target_class": "kinase",
            "druggability": 0.82,
            "source": "live",
            "confidence": 0.91,
            "timestamp": timestamp,
        }

    def fake_compounds(self: LiveAPIProvider, query: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "smiles": "CCOC(=O)N1CCC(CC1)C2=NC=CC=C2",
                "qed": 0.62,
                "source": "live",
                "confidence": 0.77,
                "chembl_id": "CHEMBL1000",
            },
            {
                "smiles": "COC1=CC=CC=C1O",
                "qed": 0.55,
                "source": "live",
                "confidence": 0.72,
                "chembl_id": "CHEMBL1001",
            },
        ]

    def fake_literature(self: LiveAPIProvider, query: str) -> list[dict[str, object]]:
        return [
            {
                "id": "pmid_1001",
                "title": "INSR signaling modulation in type 2 diabetes",
                "abstract": f"Live literature result for query: {query}",
                "year": 2024,
                "score": 0.8,
                "source": "live",
                "confidence": 0.88,
            },
            {
                "id": "pmid_1002",
                "title": "Kinase selectivity patterns and off-target toxicity",
                "abstract": "Selectivity profiling reduces adverse events.",
                "year": 2023,
                "score": 0.7,
                "source": "live",
                "confidence": 0.83,
            },
        ]

    monkeypatch.setattr(LiveAPIProvider, "get_targets_for_disease", fake_target)
    monkeypatch.setattr(LiveAPIProvider, "get_compounds", fake_compounds)
    monkeypatch.setattr(LiveAPIProvider, "search_literature", fake_literature)
