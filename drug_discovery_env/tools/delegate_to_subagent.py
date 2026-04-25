"""delegate_to_subagent — explicitly route the next review through one sub-agent."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class DelegateToSubagentTool(Tool):
    name = "delegate_to_subagent"
    default_information_gain = 0.4

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        who = str(params.get("subagent", "chemist"))
        return {
            "delegated_to": who,
            "note": "See subagent_messages on the next observation for the agent's input.",
        }
