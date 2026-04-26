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
        has_counterfactual = bool(re.search(r"if .* then|otherwise|alternative", text))
        mentions_metrics = bool(re.search(r"ic50|qed|herg|admet|selectiv|novel", text))
        raw = (
            0.18 * has_tradeoff
            + 0.18 * has_uncertainty
            + 0.18 * has_evidence
            + 0.18 * has_hypothesis
            + 0.14 * has_counterfactual
            + 0.14 * mentions_metrics
        )
        return float(raw)
