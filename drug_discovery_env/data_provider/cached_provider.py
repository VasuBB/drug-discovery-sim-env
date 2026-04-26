"""DataProvider that resolves targets from the prepared cache and proxies
compound / literature lookups to the live API provider.

This is the default mode (`data.mode: cached_targets_live_tools`). Targets are
read from `data/diseases.jsonl` so multi-disease training can rotate across
thousands of diseases without hitting Open Targets each reset, while compound
and literature retrieval stay live so the agent really exercises ChEMBL and
PubMed and learns which queries pay off.
"""

from __future__ import annotations

from typing import Any

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.dataset import (
    DiseaseDataset,
    load_dataset,
    lookup,
    now_iso,
)
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider


class CachedTargetsLiveToolsProvider(DataProvider):
    def __init__(self, settings: Settings, dataset: DiseaseDataset | None = None) -> None:
        self.settings = settings
        self._live = LiveAPIProvider(settings)
        self.dataset = dataset or load_dataset(settings.dataset.cache_path)

    def get_targets_for_disease(self, disease: str) -> dict[str, Any]:
        row = lookup(self.dataset, disease)
        if row is None:
            live = self._live.get_targets_for_disease(disease)
            live.setdefault("source", "live_fallback")
            live.setdefault("known_drugs", [])
            return live
        return {
            "disease": row.disease,
            "target": row.target,
            "target_class": row.target_class,
            "druggability": row.druggability,
            "source": "cache",
            "confidence": row.confidence,
            "timestamp": now_iso(),
            "efo_id": row.efo_id,
            "known_drugs": list(row.known_drugs),
        }

    def get_compounds(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return self._live.get_compounds(query)

    def search_literature(self, query: str) -> list[dict[str, Any]]:
        return self._live.search_literature(query)
