from __future__ import annotations

import re

from drug_discovery_env.core.models import DrugDiscoveryAction


class ReasoningReward:
    def score(self, action: DrugDiscoveryAction) -> float:
        text = action.reasoning.lower()
        has_tradeoff = bool(re.search(r"trade[- ]?off|balance|however", text))
        has_uncertainty = bool(re.search(r"uncertain|confidence|risk", text))
        has_evidence = len(action.evidence_ids) > 0
        has_hypothesis = bool(re.search(r"hypothesis|expect|predict", text))
        raw = 0.25 * has_tradeoff + 0.25 * has_uncertainty + 0.25 * has_evidence + 0.25 * has_hypothesis
        return float(raw)
