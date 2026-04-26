"""Cached disease/target dataset reader.

Reads the JSONL file produced by :mod:`drug_discovery_env.scripts.prepare_dataset`.
Each row carries the disease label, EFO id, top associated target, druggability,
known-drug SMILES (used by the eval Tanimoto metric), and a `split` flag.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from drug_discovery_env.config.runtime import resolve_path


@dataclass
class DiseaseRow:
    disease: str
    efo_id: str
    target: str
    target_class: str
    druggability: float
    confidence: float
    known_drugs: List[str] = field(default_factory=list)
    split: str = "train"


@dataclass
class DiseaseDataset:
    rows: List[DiseaseRow]
    by_disease: Dict[str, DiseaseRow]
    train: List[DiseaseRow]
    test: List[DiseaseRow]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


_CACHE: Dict[str, DiseaseDataset] = {}
_CACHE_LOCK = threading.Lock()


def _key_disease(name: str) -> str:
    return name.strip().lower()


def load_dataset(path: str | Path) -> DiseaseDataset:
    resolved = resolve_path(path)
    cache_key = str(resolved)
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]
    if not resolved.exists():
        raise FileNotFoundError(
            f"Disease dataset not found at {resolved}. "
            "Run `python -m drug_discovery_env.scripts.prepare_dataset` first."
        )
    rows: List[DiseaseRow] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                DiseaseRow(
                    disease=str(payload["disease"]),
                    efo_id=str(payload.get("efo_id", "")),
                    target=str(payload["target"]),
                    target_class=str(payload.get("target_class", "unknown")),
                    druggability=float(payload.get("druggability", 0.0)),
                    confidence=float(payload.get("confidence", 0.0)),
                    known_drugs=list(payload.get("known_drugs", [])),
                    split=str(payload.get("split", "train")),
                )
            )
    by_disease = {_key_disease(r.disease): r for r in rows}
    train = [r for r in rows if r.split == "train"]
    test = [r for r in rows if r.split == "test"]
    dataset = DiseaseDataset(rows=rows, by_disease=by_disease, train=train, test=test)
    with _CACHE_LOCK:
        _CACHE[cache_key] = dataset
    return dataset


def lookup(dataset: DiseaseDataset, disease: str) -> Optional[DiseaseRow]:
    return dataset.by_disease.get(_key_disease(disease))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
