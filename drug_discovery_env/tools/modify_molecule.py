"""modify_molecule — RDKit-validated SMILES edits (methyl, fluorine, etc.)."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class ModifyMoleculeTool(Tool):
    name = "modify_molecule"
    default_information_gain = 0.5

    def __init__(self, settings: Settings, lab: LabSimulator | None = None) -> None:
        self.settings = settings
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        smiles = params.get("smiles")
        compound_id = params.get("compound_id") or params.get("target_compound_id")
        if not smiles and compound_id:
            rec = state.get_by_id(str(compound_id))
            smiles = rec.smiles if rec else None
        if not smiles:
            return {"error": "no_smiles", "message": "modify_molecule requires `smiles` or a compound_id"}

        instruction = str(params.get("instruction", "add methyl"))
        mod = self.lab.modify_molecule(smiles, instruction)
        new_smiles = mod["smiles"]

        parent = state.compound_ledger.get(smiles)
        rec = state.add_compound(
            new_smiles,
            origin="modified",
            parent_id=parent.id if parent else None,
            history=f"modify({instruction})",
        )
        # Inherit + perturb scores from parent so the variant has a head start
        if parent:
            rec.potency = max(0.0, min(1.0, parent.potency * 0.9))
            rec.selectivity = max(0.0, min(1.0, parent.selectivity * 1.05))
            rec.developability = max(0.0, min(1.0, parent.developability * 1.05))
            rec.uncertainty = max(0.05, parent.uncertainty - 0.05)
            rec.novelty = min(1.0, parent.novelty + 0.1)
            rec.metadata["parent"] = parent.smiles
        rec.metadata["strategy"] = instruction
        return {
            "parent_smiles": smiles,
            "child_smiles": new_smiles,
            "child_compound_id": rec.id,
            "instruction": instruction,
        }
