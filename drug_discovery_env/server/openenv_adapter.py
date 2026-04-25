from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.server.environment import DrugDiscoveryEnv


@dataclass
class StepResult:
    observation: DrugDiscoveryObservation
    reward: float
    done: bool
    info: dict[str, Any]


class OpenEnvEnvironmentAdapter:
    """Compatibility adapter for OpenEnv-style runtimes.

    If openenv-core classes are available, this adapter can be wrapped or subclassed
    by the concrete runtime integration without changing environment logic.
    """

    def __init__(self, env: DrugDiscoveryEnv | None = None) -> None:
        self.env = env or DrugDiscoveryEnv()

    def reset(self) -> DrugDiscoveryObservation:
        return self.env.reset()

    def step(self, action: DrugDiscoveryAction | str) -> StepResult:
        obs = self.env.step(action)
        total_reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
        return StepResult(observation=obs, reward=total_reward, done=obs.done, info=obs.info)


def create_openenv_adapter() -> OpenEnvEnvironmentAdapter:
    return OpenEnvEnvironmentAdapter()
