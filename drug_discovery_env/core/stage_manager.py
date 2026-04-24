from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


class StageManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def update_stage(self, state: GameState) -> int:
        gates = self.settings.transitions.stage_gates
        hits = sum(1 for c in state.compound_ledger.values() if c.potency > 0.4)
        optimized = sum(1 for c in state.compound_ledger.values() if c.developability > 0.5)
        admet_ready = sum(1 for c in state.compound_ledger.values() if c.safety > 0.5)

        if state.stage == 1 and state.target:
            state.stage = 2
        elif state.stage == 2 and hits >= int(gates["stage_2_min_hits"]):
            state.stage = 3
        elif state.stage == 3 and optimized >= int(gates["stage_3_min_optimized"]):
            state.stage = 4
        elif state.stage == 4 and admet_ready >= int(gates["stage_4_min_admet"]):
            state.stage = 5
        return state.stage
