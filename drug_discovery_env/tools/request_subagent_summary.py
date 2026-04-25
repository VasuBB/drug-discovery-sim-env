"""request_subagent_summary — pull the full inbox of recent sub-agent advice."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class RequestSubagentSummaryTool(Tool):
    name = "request_subagent_summary"
    default_information_gain = 0.4

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": dict(state.sub_agent_inbox),
            "open_block_warnings": list(state.open_block_warnings),
        }
