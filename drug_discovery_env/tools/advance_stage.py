"""advance_stage — agent-driven progression through the 5-stage pipeline."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.models import STAGE_ORDER
from drug_discovery_env.core.stage_manager import StageManager
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class AdvanceStageTool(Tool):
    name = "advance_stage"
    default_information_gain = 0.9  # high-info: it's a real strategic decision

    def __init__(self, settings: Settings, stage_manager: StageManager) -> None:
        self.settings = settings
        self.stage_manager = stage_manager

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        current = state.stage

        # target_selection: optionally accept the selected target inline
        if current == "target_selection":
            target = params.get("target")
            if target and not state.selected_target:
                state.selected_target = str(target)
                state.target = state.target or {
                    "disease": state.disease,
                    "target": str(target),
                    "target_class": "agent_specified",
                    "druggability": 0.5,
                    "source": "agent_specified",
                    "confidence": 0.6,
                }

        ok, msg, next_stage = self.stage_manager.can_advance(state)
        if not ok:
            return {"advanced": False, "stage": current, "reason": msg}

        # lead_validation: nominate a lead
        if current == "lead_validation":
            cid = params.get("compound_id") or params.get("target_compound_id")
            rec = state.get_by_id(str(cid)) if cid else state.best_compound()
            if rec is None:
                return {
                    "advanced": False,
                    "stage": current,
                    "reason": "no compound to nominate as lead",
                }
            state.advanced_compound_id = rec.id
            state.terminated_reason = "completed"

        state.stages_completed.append(current)
        state.stage = next_stage or current
        return {
            "advanced": True,
            "previous_stage": current,
            "stage": state.stage,
            "stages_completed": list(state.stages_completed),
            "nominated_lead_id": state.advanced_compound_id,
            "is_finished": state.stage == "finished",
            "stage_index": STAGE_ORDER.index(state.stage),
        }
