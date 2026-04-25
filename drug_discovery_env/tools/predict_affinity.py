"""predict_affinity — RDKit-grounded binding affinity prediction."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.chemistry.rdkit_lab import LabSimulator
from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState
from drug_discovery_env.core.topology import TopologyEngine
from drug_discovery_env.tools.base import Tool


class PredictAffinityTool(Tool):
    name = "predict_affinity"
    default_information_gain = 0.7

    def __init__(self, settings: Settings, topology: TopologyEngine, lab: LabSimulator | None = None) -> None:
        self.settings = settings
        self.topology = topology
        self.lab = lab or LabSimulator()

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        smiles = params.get("smiles")
        compound_id = params.get("compound_id") or params.get("target_compound_id")
        if not smiles and compound_id:
            rec = state.get_by_id(str(compound_id))
            smiles = rec.smiles if rec else None
        if not smiles:
            return {"error": "no_smiles", "message": "predict_affinity requires `smiles` or a compound_id"}
        if smiles not in state.compound_ledger:
            state.add_compound(smiles, origin="agent_specified")

        target = params.get("target") or state.selected_target
        if not target:
            return {"error": "no_target", "message": "Select a target before predicting affinity"}

        result = self.lab.predict_binding_affinity(smiles, str(target))
        rec = state.compound_ledger[smiles]
        rec.binding_affinity_nM = float(result["affinity_nM"])
        rec.potency = float(result["score"])
        rec.assay_count += 1
        rec.uncertainty = max(0.05, rec.uncertainty - 0.1)

        # Topology-aware off-target risk propagation
        self.topology.propagate_risk(state, str(target), base_risk=1.0 - rec.potency)

        return {
            "smiles": smiles,
            "compound_id": rec.id,
            "target": target,
            "affinity_nM": rec.binding_affinity_nM,
            "score": rec.potency,
            "uncertainty": rec.uncertainty,
        }
