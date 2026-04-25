"""Typed OpenEnv client for the Drug Discovery Sim Environment.

Talks to a running server over HTTP/WebSocket; does not import server
internals so client/server separation is preserved. Use as:

    from drug_discovery_env.client import DrugDiscoveryClient
    env = DrugDiscoveryClient(base_url="http://localhost:8000").sync()
    res = env.reset()
    res = env.step(DrugDiscoveryAction(tool="select_target", params={"target": "DPP4"}))
"""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.models import (
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
)
from drug_discovery_env.openenv_compat import EnvClient

try:
    from openenv.core.client_types import StepResult
except Exception:  # pragma: no cover — legacy fallback
    from openenv_core.client_types import StepResult  # type: ignore


class DrugDiscoveryClient(EnvClient[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    def _step_payload(self, action: DrugDiscoveryAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DrugDiscoveryObservation]:
        obs_payload = payload.get("observation", payload) or {}
        observation = DrugDiscoveryObservation.model_validate(obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_payload.get("reward")),
            done=payload.get("done", obs_payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DrugDiscoveryState:
        return DrugDiscoveryState.model_validate(payload)


def create_sync_client(base_url: str):
    return DrugDiscoveryClient(base_url=base_url).sync()
