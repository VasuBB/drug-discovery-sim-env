from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import CompoundRecord


class TerminalReward:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def score(self, compound: CompoundRecord | None) -> float:
        if not compound:
            return 0.0
        floors = self.settings.reward.floors
        admet = compound.metadata.get("admet", {})
        if float(admet.get("herg_prob", 0.0)) > float(floors["herg_prob_max"]):
            return 0.0

        components = self.settings.reward.terminal_components
        values = {
            "potency": compound.potency,
            "selectivity": compound.selectivity,
            "safety": compound.safety,
            "synthesizability": compound.synthesizability,
            "novelty": compound.novelty,
            "developability": compound.developability,
        }
        total = sum(float(components[k]) * max(0.0, min(1.0, values[k])) for k in components)
        return max(0.0, min(1.0, total))
