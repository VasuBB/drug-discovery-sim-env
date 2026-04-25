"""Client for the Drug Discovery Sim Environment.

Importable in training code as:

    from client import DrugDiscoveryEnv
    env = DrugDiscoveryEnv(base_url="http://localhost:8000").sync()
    result = env.reset()
    result = env.step(DrugDiscoveryAction(tool="search_chembl", params={"query": "DPP4"}))
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState


class DrugDiscoveryEnv(EnvClient[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    """Type-safe HTTP/WebSocket client for the drug discovery campaign env."""

    def _step_payload(self, action: DrugDiscoveryAction) -> Dict[str, Any]:
        return {
            "tool": action.tool,
            "params": action.params or {},
            "target_compound_id": action.target_compound_id,
            "reasoning": action.reasoning or "",
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", payload) or {}
        obs = DrugDiscoveryObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            stage=obs_data.get("stage", "target_selection"),
            step_index=obs_data.get("step_index", 0),
            max_steps=obs_data.get("max_steps", 50),
            disease=obs_data.get("disease", ""),
            selected_target=obs_data.get("selected_target"),
            budget_total=obs_data.get("budget_total", 0.0),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            last_tool_cost=obs_data.get("last_tool_cost", 0.0),
            last_tool=obs_data.get("last_tool"),
            last_result=obs_data.get("last_result", {}),
            active_compounds=obs_data.get("active_compounds", []),
            advanced_compound_id=obs_data.get("advanced_compound_id"),
            subagent_messages=obs_data.get("subagent_messages", []),
            message=obs_data.get("message", ""),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DrugDiscoveryState:
        return DrugDiscoveryState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            disease=payload.get("disease", ""),
            selected_target=payload.get("selected_target"),
            max_steps=payload.get("max_steps", 50),
            budget_total=payload.get("budget_total", 1000.0),
            budget_remaining=payload.get("budget_remaining", 1000.0),
            stage=payload.get("stage", "target_selection"),
            advanced_compound_id=payload.get("advanced_compound_id"),
            terminated_reason=payload.get("terminated_reason"),
        )
