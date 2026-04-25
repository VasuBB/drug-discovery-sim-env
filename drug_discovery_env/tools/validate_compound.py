"""validate_compound — final docking + selectivity panel (RDKit-flavoured)."""

from __future__ import annotations

import random
from typing import Any, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class ValidateCompoundTool(Tool):
    name = "validate_compound"
    default_information_gain = 0.8

    def __init__(self, lab: LabSimulator | None = None) -> None:
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        smiles = params.get("smiles")
        compound_id = params.get("compound_id") or params.get("target_compound_id")
        if not smiles and compound_id:
            rec = state.get_by_id(str(compound_id))
            smiles = rec.smiles if rec else None
        if not smiles:
            return {"error": "no_smiles", "message": "validate_compound requires `smiles` or a compound_id"}
        if smiles not in state.compound_ledger:
            state.add_compound(smiles, origin="agent_specified")

        target = params.get("target") or state.selected_target or "generic"
        panel = list(params.get("panel", ["hERG", "CYP3A4", "5HT2B"]))

        docking = self.lab.run_docking(smiles, str(target))
        rec = state.compound_ledger[smiles]
        rec.docking_score = float(docking["docking_score"])

        selectivity = max(0.0, min(1.0, rec.selectivity + random.gauss(0, 0.07)))
        rec.selectivity = selectivity
        off_target_hits = {p: max(0.0, min(1.0, random.gauss(0.35, 0.2))) for p in panel}
        rec.metadata["validation"] = {
            "docking": docking,
            "panel": off_target_hits,
        }
        return {
            "smiles": smiles,
            "compound_id": rec.id,
            "target": target,
            "docking_score": rec.docking_score,
            "docking_quality": docking["quality"],
            "selectivity": selectivity,
            "off_target_hits": off_target_hits,
        }
