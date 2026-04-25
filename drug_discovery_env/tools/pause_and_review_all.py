"""pause_and_review_all — solicit a fresh sub-agent panel review without spending."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class PauseAndReviewAllTool(Tool):
    name = "pause_and_review_all"
    default_information_gain = 0.4

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paused": True,
            "active_count": len(state.compound_ledger),
            "stage": state.stage,
            "budget_remaining": state.budget_remaining,
        }
