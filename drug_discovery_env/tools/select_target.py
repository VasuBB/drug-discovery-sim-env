"""select_target — pick a protein target for the disease.

Backed by the data provider (Open Targets live with local fallback) and falls
back to the built-in scenario library when no provider hit is found.
"""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.scenarios import find_scenario
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class SelectTargetTool(Tool):
    name = "select_target"
    default_information_gain = 0.6

    def __init__(self, settings: Settings, provider: Any) -> None:
        self.settings = settings
        self.provider = provider

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        explicit = params.get("target")
        disease = params.get("disease", state.disease)

        if explicit:
            target = {
                "disease": disease,
                "target": str(explicit),
                "target_class": str(params.get("target_class", "unknown")),
                "druggability": float(params.get("druggability", 0.5)),
                "source": "agent_specified",
                "confidence": 0.7,
            }
        else:
            try:
                target = self.provider.get_targets_for_disease(disease)
            except Exception:
                target = {}
            if not target.get("target") or target.get("target") == "UNKNOWN_TARGET":
                fallback = find_scenario(disease)
                if fallback:
                    target = {
                        "disease": disease,
                        "target": fallback.canonical_target,
                        "target_class": "scenario",
                        "druggability": 1.0 - fallback.difficulty,
                        "source": "simulation",
                        "confidence": 0.6,
                    }

        state.target = target
        state.selected_target = target.get("target")
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
