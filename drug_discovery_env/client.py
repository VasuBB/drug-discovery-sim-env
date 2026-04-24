from __future__ import annotations

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.server.environment import DrugDiscoveryEnv


class DrugDiscoveryClient:
    def __init__(self, env: DrugDiscoveryEnv | None = None) -> None:
        self.env = env or DrugDiscoveryEnv()

    def reset(self, disease: str = "Type 2 Diabetes") -> DrugDiscoveryObservation:
        return self.env.reset(disease=disease)

    def step(self, action: DrugDiscoveryAction | str) -> DrugDiscoveryObservation:
        return self.env.step(action)
