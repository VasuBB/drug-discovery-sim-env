"""Stage gate checker — does NOT auto-advance.

The agent decides when to move stages by calling the `advance_stage` tool.
This module enforces preconditions for that transition (e.g. "can't leave
hit_id with zero compounds"). On success it returns the next stage name; on
failure it returns a human-readable reason that surfaces back to the agent.
"""

from __future__ import annotations

from typing import Optional, Tuple

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.models import STAGE_ORDER
from drug_discovery_env.core.state import GameState


class StageManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_advance(self, state: GameState) -> Tuple[bool, str, Optional[str]]:
        """Return (allowed, reason_or_next_stage, next_stage)."""
        current = state.stage
        idx = STAGE_ORDER.index(current) if current in STAGE_ORDER else -1
        if idx < 0 or current == "finished":
            return False, "already_finished", None

        gates = self.settings.transitions.stage_gates
        ledger = state.compound_ledger.values()

        if current == "target_selection":
            if not state.selected_target:
                return False, "must select a target before advancing from target_selection", None
            return True, "ok", STAGE_ORDER[idx + 1]

        if current == "hit_id":
            min_hits = int(gates.get("stage_2_min_hits", 1))
            if len(state.compound_ledger) < min_hits:
                return False, f"need at least {min_hits} compound(s) to advance from hit_id", None
            return True, "ok", STAGE_ORDER[idx + 1]

        if current == "hit_to_lead":
            min_opt = int(gates.get("stage_3_min_optimized", 1))
            optimized = sum(
                1 for c in ledger
                if (c.binding_affinity_nM is not None) or (c.potency > 0.0) or c.history
            )
            if optimized < min_opt:
                return False, f"need at least {min_opt} optimized compound(s) to advance from hit_to_lead", None
            return True, "ok", STAGE_ORDER[idx + 1]

        if current == "admet":
            min_admet = int(gates.get("stage_4_min_admet", 1))
            with_admet = sum(1 for c in ledger if c.admet)
            if with_admet < min_admet:
                return False, f"need ADMET data on at least {min_admet} compound(s) to advance from admet", None
            return True, "ok", STAGE_ORDER[idx + 1]

        if current == "lead_validation":
            return True, "ok", "finished"

        return False, f"unknown stage {current}", None
