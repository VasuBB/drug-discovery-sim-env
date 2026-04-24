from __future__ import annotations

import random

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import CompoundRecord, GameState
from drug_discovery_env.tools.base import Tool


class SynthesizeTool(Tool):
    name = "synthesize"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        a = str(params.get("smiles_a", ""))
        b = str(params.get("smiles_b", ""))
        route = str(params.get("route", "amide_coupling"))
        if a not in state.compound_ledger or b not in state.compound_ledger:
            raise ValueError("Unknown reactants")

        fail_prob = self.settings.tools.synthesis_failure_base
        if route in {"buchwald_hartwig", "suzuki_coupling"}:
            fail_prob += 0.06
        if random.random() < fail_prob:
            return {"success": False, "route": route, "reason": "route_failure"}

        child = f"{a}|{b}|S"
        state.compound_ledger[child] = CompoundRecord(
            smiles=child,
            potency=max(state.compound_ledger[a].potency, state.compound_ledger[b].potency),
            selectivity=(state.compound_ledger[a].selectivity + state.compound_ledger[b].selectivity) / 2,
            safety=(state.compound_ledger[a].safety + state.compound_ledger[b].safety) / 2,
            synthesizability=max(0.0, min(1.0, 0.65 - fail_prob)),
            novelty=0.7,
            developability=0.55,
            uncertainty=0.5,
            metadata={"route": route},
        )
        return {"success": True, "product": child, "route": route}
