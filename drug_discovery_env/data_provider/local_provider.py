from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.snapshot_utils import (
    validate_manifest,
    validate_snapshot_payloads,
)
from drug_discovery_env.scripts.prepare_snapshots import main as prepare_manifest_main


class LocalSnapshotProvider(DataProvider):
    def __init__(self, local_data_dir: str) -> None:
        self.root = Path(local_data_dir)
        if (self.root / "snapshot_manifest.json").exists():
            try:
                validate_manifest(self.root)
            except Exception:
                # Self-heal when snapshot files changed since last manifest generation.
                prepare_manifest_main()
                validate_manifest(self.root)
        validate_snapshot_payloads(self.root)

    def _load_json(self, filename: str, default: Any) -> Any:
        path = self.root / filename
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_targets_for_disease(self, disease: str) -> dict[str, Any]:
        rows = self._load_json("targets_snapshot.json", [])
        candidates = [r for r in rows if r.get("disease", "").lower() == disease.lower()]
        if not candidates:
            return {
                "disease": disease,
                "target": "UNKNOWN1",
                "target_class": "unknown",
                "druggability": 0.3,
                "source": "local",
                "confidence": 0.3,
            }
        best = max(candidates, key=lambda x: x.get("druggability", 0.0))
        best["source"] = "local"
        return best

    def get_compounds(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        library = self._load_json("compound_library.json", [])
        min_qed = float(query.get("min_qed", 0.0))
        return [c for c in library if float(c.get("qed", 0.0)) >= min_qed][:30]

    def search_literature(self, query: str) -> list[dict[str, Any]]:
        docs = self._load_json("literature_snapshot.json", [])
        query_tokens = {t.lower() for t in query.split()}
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}".lower()
            overlap = len(query_tokens.intersection(set(text.split())))
            scored.append((float(overlap), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, doc in scored[:10]:
            row = dict(doc)
            row["score"] = score
            row["source"] = "local"
            out.append(row)
        return out
