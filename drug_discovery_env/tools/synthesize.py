"""synthesize — combine two parent compounds via a named reaction route."""

from __future__ import annotations

import random
from typing import Any, Dict

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class SynthesizeTool(Tool):
    name = "synthesize"
    default_information_gain = 0.4

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        a = str(params.get("smiles_a", ""))
        b = str(params.get("smiles_b", ""))
        route = str(params.get("route", "amide_coupling"))
        if a not in state.compound_ledger or b not in state.compound_ledger:
            return {"success": False, "error": "unknown_reactants", "route": route}

        fail_prob = self.settings.tools.synthesis_failure_base
        if route in {"buchwald_hartwig", "suzuki_coupling"}:
            fail_prob += 0.06
        if random.random() < fail_prob:
            return {"success": False, "route": route, "reason": "route_failure"}

        child_smiles = f"{a}|{b}|S"
        rec_a = state.compound_ledger[a]
        rec_b = state.compound_ledger[b]
        rec = state.add_compound(
            child_smiles,
            origin="synthesized",
            parent_id=rec_a.id,
            potency=max(rec_a.potency, rec_b.potency),
            selectivity=(rec_a.selectivity + rec_b.selectivity) / 2,
            safety=(rec_a.safety + rec_b.safety) / 2,
            synthesizability=max(0.0, min(1.0, 0.65 - fail_prob)),
            novelty=0.7,
            developability=0.55,
            uncertainty=0.5,
        )
        rec.metadata["route"] = route
        return {
            "success": True,
            "product_smiles": child_smiles,
            "product_compound_id": rec.id,
            "route": route,
        }
