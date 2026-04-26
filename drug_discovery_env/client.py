"""Typed OpenEnv client for the Drug Discovery Sim Environment.

Talks to a running server over HTTP; does not import server internals so
client/server separation is preserved.

Example:

    from drug_discovery_env.client import DrugDiscoveryClient
    from drug_discovery_env.core.models import DrugDiscoveryAction

    with DrugDiscoveryClient(base_url="https://vasuboda-drug-discovery-sim-env.hf.space").sync() as env:
        res = env.reset(disease="Type 2 Diabetes")
        res = env.step(DrugDiscoveryAction(
            tool="select_target",
            params={"disease": "Type 2 Diabetes"},
            reasoning="start",
        ))
        print(res.observation, res.reward, res.done)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

from drug_discovery_env.core.models import (
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
)
from drug_discovery_env.openenv_compat import EnvClient

# Prefer the real openenv StepResult if available; otherwise fall back to a
# local dataclass with the same shape (.observation / .reward / .done).
try:
    from openenv.core.client_types import StepResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from openenv_core.client_types import StepResult  # type: ignore
    except Exception:
        O = TypeVar("O")

        @dataclass
        class StepResult(Generic[O]):  # type: ignore[no-redef]
            observation: O
            reward: float | None
            done: bool


class DrugDiscoveryClient(EnvClient[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    def _step_payload(self, action: DrugDiscoveryAction) -> Dict[str, Any]:
        if isinstance(action, DrugDiscoveryAction):
            return action.model_dump()
        if isinstance(action, dict):
            return dict(action)
        return {"tool": str(action)}

    def _parse_result(self, payload: Dict[str, Any]) -> "StepResult[DrugDiscoveryObservation]":
        obs_payload = payload.get("observation", payload) or {}
        observation = DrugDiscoveryObservation.model_validate(obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_payload.get("reward")),
            done=bool(payload.get("done", obs_payload.get("done", False))),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DrugDiscoveryState:
        return DrugDiscoveryState.model_validate(payload)


def create_sync_client(base_url: str) -> DrugDiscoveryClient:
    return DrugDiscoveryClient(base_url=base_url).sync()
