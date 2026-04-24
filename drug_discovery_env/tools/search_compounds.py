from __future__ import annotations

from typing import Any

from drug_discovery_env.core.state import CompoundRecord, GameState
from drug_discovery_env.tools.base import Tool


class SearchCompoundsTool(Tool):
    name = "search_compounds"

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def execute(self, state: GameState, params: dict[str, Any]) -> dict[str, Any]:
        min_qed = float(params.get("min_qed", 0.4))
        compounds = self.provider.get_compounds({"min_qed": min_qed, "target": state.target})
        for c in compounds[:20]:
            smiles = c.get("smiles", "")
            if smiles and smiles not in state.compound_ledger:
                state.compound_ledger[smiles] = CompoundRecord(
                    smiles=smiles,
                    developability=float(c.get("qed", 0.0)),
                    uncertainty=0.8,
                )
        return {"hits": compounds[:20], "count": len(compounds[:20])}
