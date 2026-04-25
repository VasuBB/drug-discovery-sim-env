"""search_compounds — populate the compound pool.

Tries the data provider first; on empty result falls back to RDKit lab's
default library so the env still produces hits offline.
"""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class SearchCompoundsTool(Tool):
    name = "search_compounds"
    default_information_gain = 0.5

    def __init__(self, provider: Any, lab: LabSimulator | None = None) -> None:
        self.provider = provider
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        min_qed = float(params.get("min_qed", 0.4))
        query = params.get("query") or state.selected_target or state.disease

        compounds: list[dict[str, Any]] = []
        try:
            compounds = self.provider.get_compounds({"min_qed": min_qed, "target": state.target})
        except Exception:
            compounds = []

        if not compounds:
            seeds = self.lab.search_chembl(query=query, max_results=int(params.get("max_results", 5)))
            compounds = [
                {"smiles": s["smiles"], "qed": 0.55, "source": "simulation", "confidence": 0.5}
                for s in seeds
            ]

        added = []
        for c in compounds[:20]:
            smiles = c.get("smiles", "")
            if not smiles:
                continue
            rec = state.add_compound(
                smiles,
                origin=str(c.get("source", "search")),
                developability=float(c.get("qed", 0.0)),
                uncertainty=0.8,
            )
            added.append({"id": rec.id, "smiles": smiles, "qed": c.get("qed")})
        return {"hits": compounds[:20], "added": added, "count": len(added)}
