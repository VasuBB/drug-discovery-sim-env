from __future__ import annotations

from typing import Any

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class SelectTargetTool(Tool):
    name = "select_target"

    def __init__(self, settings: Settings, provider: Any) -> None:
        self.settings = settings
        self.provider = provider

    def execute(self, state: GameState, params: dict[str, Any]) -> dict[str, Any]:
        disease = params.get("disease", state.disease)
        target = self.provider.get_targets_for_disease(disease)
        state.target = target
        state.target_class = target.get("target_class", "unknown")
        state.assay_confidence["target_selection"] = float(target.get("confidence", 0.5))
        return {
            "target": target,
            "provenance": {
                "source": target.get("source", "unknown"),
                "timestamp": target.get("timestamp", ""),
                "confidence": target.get("confidence", 0.5),
            },
        }
