from __future__ import annotations

import random

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.topology import TopologyEngine
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class PredictAffinityTool(Tool):
    name = "predict_affinity"

    def __init__(self, settings: Settings, topology: TopologyEngine) -> None:
        self.settings = settings
        self.topology = topology

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        smiles = str(params.get("smiles", ""))
        assay = str(params.get("assay_type", "biochemical"))
        if smiles not in state.compound_ledger:
            raise ValueError("Unknown compound")
        record = state.compound_ledger[smiles]
        noise = random.gauss(0.0, self.settings.tools.assay_noise.get(assay, 0.15))
        base = 0.55 + noise - 0.15 * record.uncertainty
        potency = max(0.0, min(1.0, base))
        record.potency = potency
        record.assay_count += 1
        record.uncertainty = max(0.05, record.uncertainty - 0.1)

        if state.target and state.target.get("target"):
            self.topology.propagate_risk(state, state.target["target"], base_risk=1.0 - potency)

        return {
            "smiles": smiles,
            "assay_type": assay,
            "predicted_potency": potency,
            "uncertainty": record.uncertainty,
        }
