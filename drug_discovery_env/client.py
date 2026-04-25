from __future__ import annotations

from typing import Any

from drug_discovery_env.openenv_compat import EnvClient
try:
    from openenv.core.client_types import StepResult
except Exception:  # pragma: no cover - legacy path
    from openenv_core.client_types import StepResult

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState


class DrugDiscoveryClient(EnvClient[DrugDiscoveryAction, DrugDiscoveryObservation, DrugDiscoveryState]):
    """Typed OpenEnv client for the drug discovery environment.

    This client talks to a running server over WebSocket/HTTP and does not import
    server internals, preserving client/server separation.
    """

    def _step_payload(self, action: DrugDiscoveryAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[DrugDiscoveryObservation]:
        obs_payload = payload.get("observation", {})
        observation = DrugDiscoveryObservation.model_validate(obs_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> DrugDiscoveryState:
        return DrugDiscoveryState.model_validate(payload)


def create_sync_client(base_url: str):
    return DrugDiscoveryClient(base_url=base_url).sync()
