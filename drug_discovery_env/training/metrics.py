"""Evaluation metrics — env reward + ChEMBL Tanimoto + ADMET / oversight panel.

Per-test-disease record is a flat dict; aggregate is computed by
:func:`aggregate_report`. Tanimoto uses RDKit Morgan fingerprints when RDKit
is available and falls back to a canonical-string Jaccard score (logged once)
otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from drug_discovery_env.config.runtime import resolve_path

logger = logging.getLogger(__name__)

_RDKIT_WARNED = False


def _rdkit_morgan(smiles: str, radius: int, n_bits: int):  # pragma: no cover
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(a: str, b: str, radius: int = 2, n_bits: int = 2048) -> float:
    if not a or not b:
        return 0.0
    try:
        from rdkit import DataStructs

        fa = _rdkit_morgan(a, radius, n_bits)
        fb = _rdkit_morgan(b, radius, n_bits)
        if fa is None or fb is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(fa, fb))
    except Exception:
        global _RDKIT_WARNED
        if not _RDKIT_WARNED:
            logger.warning("RDKit unavailable; Tanimoto falls back to canonical-string Jaccard.")
            _RDKIT_WARNED = True
        sa = set(a)
        sb = set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


def best_tanimoto(candidate: str, references: Iterable[str], radius: int, n_bits: int) -> float:
    best = 0.0
    for ref in references:
        score = tanimoto(candidate, ref, radius=radius, n_bits=n_bits)
        if score > best:
            best = score
    return best


def mean_tanimoto(candidate: str, references: Iterable[str], radius: int, n_bits: int) -> float:
    refs = [r for r in references if r]
    if not refs:
        return 0.0
    return sum(tanimoto(candidate, r, radius=radius, n_bits=n_bits) for r in refs) / len(refs)


@dataclass
class DiseaseMetricRecord:
    disease: str
    target: str
    nominated_smiles: Optional[str]
    terminal_reward: float
    total_reward: float
    reward_breakdown: Dict[str, float]
    final_stage_idx: int
    stage_completed: bool
    budget_remaining_frac: float
    admet_pass: bool
    oversight_violations: int
    mean_reasoning_depth: float
    tanimoto_to_known_max: float
    tanimoto_to_known_mean: float
    precision_at_1: float
    n_known_drugs: int
    steps: int
    terminated_reason: str
    extras: Dict[str, Any] = field(default_factory=dict)


def write_per_disease(records: List[DiseaseMetricRecord], path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(asdict(rec), default=str) + "\n")
    return out


def aggregate_report(records: List[DiseaseMetricRecord], per_disease_path: Path) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_test_diseases": 0, "per_disease_path": str(per_disease_path)}

    def _mean(values: List[float]) -> float:
        return sum(values) / max(1, len(values))

    return {
        "n_test_diseases": n,
        "mean_terminal_reward": _mean([r.terminal_reward for r in records]),
        "mean_total_reward": _mean([r.total_reward for r in records]),
        "stage_completion_rate": _mean([1.0 if r.stage_completed else 0.0 for r in records]),
        "admet_pass_rate": _mean([1.0 if r.admet_pass else 0.0 for r in records]),
        "oversight_violation_rate": _mean(
            [1.0 if r.oversight_violations > 0 else 0.0 for r in records]
        ),
        "mean_oversight_violations": _mean([float(r.oversight_violations) for r in records]),
        "mean_budget_remaining_frac": _mean([r.budget_remaining_frac for r in records]),
        "mean_reasoning_depth": _mean([r.mean_reasoning_depth for r in records]),
        "mean_tanimoto_to_known": _mean([r.tanimoto_to_known_mean for r in records]),
        "mean_tanimoto_to_known_max": _mean([r.tanimoto_to_known_max for r in records]),
        "precision_at_1": _mean([r.precision_at_1 for r in records]),
        "mean_steps": _mean([float(r.steps) for r in records]),
        "per_disease_path": str(per_disease_path),
    }
