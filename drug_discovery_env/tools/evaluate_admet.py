"""evaluate_admet — RDKit-backed ADMET evaluation (PAINS, Lipinski, tox, QED)."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class EvaluateAdmetTool(Tool):
    name = "evaluate_admet"
    default_information_gain = 0.7

    def __init__(self, lab: LabSimulator | None = None) -> None:
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        smiles = params.get("smiles")
        compound_id = params.get("compound_id") or params.get("target_compound_id")
        if not smiles and compound_id:
            rec = state.get_by_id(str(compound_id))
            smiles = rec.smiles if rec else None
        if not smiles:
            return {"error": "no_smiles", "message": "evaluate_admet requires `smiles` or a compound_id"}
        if smiles not in state.compound_ledger:
            state.add_compound(smiles, origin="agent_specified")

        admet = self.lab.compute_admet(smiles).to_dict()
        rec = state.compound_ledger[smiles]
        rec.admet = admet
        rec.metadata["admet"] = {
            "herg_prob": 1.0 if admet["pains"] else min(0.99, max(0.0, admet["tox_score"])),
            "caco2": max(0.0, min(1.0, 0.7 - 0.5 * admet["tox_score"])),
            "tox": admet["tox_score"],
        }
        # Safety surrogate combining ADMET signals
        safety = 0.0
        if admet["ro5_pass"]:
            safety += 0.5
        if not admet["pains"]:
            safety += 0.25
        if not admet["tox_flag"]:
            safety += 0.25
        rec.safety = safety
        return {
            "smiles": smiles,
            "compound_id": rec.id,
            "admet": admet,
            "safety": safety,
        }
