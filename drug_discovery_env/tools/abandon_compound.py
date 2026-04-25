"""abandon_compound — drop a compound from the active pool."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class AbandonCompoundTool(Tool):
    name = "abandon_compound"
    default_information_gain = 0.3

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        cid = params.get("compound_id") or params.get("target_compound_id")
        smiles = params.get("smiles")

        rec = None
        if cid:
            rec = state.get_by_id(str(cid))
        if rec is None and smiles:
            rec = state.compound_ledger.get(str(smiles))
        if rec is None:
            return {"error": "no_such_compound", "compound_id": cid, "smiles": smiles}

        state.compound_ledger.pop(rec.smiles, None)
        if rec.id:
            state.compound_id_index.pop(rec.id, None)
        if state.advanced_compound_id == rec.id:
            state.advanced_compound_id = None
        return {"abandoned_id": rec.id, "abandoned_smiles": rec.smiles}
