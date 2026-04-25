from __future__ import annotations

import re

from drug_discovery_env.core.models import DrugDiscoveryAction


SCIENTIFIC_KEYWORDS = (
    "binding", "affinity", "admet", "toxic", "pains", "lipinski",
    "scaffold", "selectivity", "potency", "docking", "novelty",
    "budget", "trade-off", "tradeoff", "hypothesis", "off-target",
    "modification", "candidate", "screen", "lead", "herg",
)


class ReasoningReward:
    """Mercor-style reasoning reward.

    Combines a structural signal (mentions evidence, weighs trade-offs) with a
    scientific-keyword density signal ported from the `main` branch.
    """

    def score(self, action: DrugDiscoveryAction) -> float:
        text = (action.reasoning or "").lower()
        has_tradeoff = bool(re.search(r"trade[- ]?off|balance|however", text))
        has_uncertainty = bool(re.search(r"uncertain|confidence|risk", text))
        has_evidence = len(action.evidence_ids) > 0
        has_hypothesis = bool(re.search(r"hypothesis|expect|predict", text))
        structural = 0.25 * has_tradeoff + 0.25 * has_uncertainty + 0.25 * has_evidence + 0.25 * has_hypothesis

        if not text:
            keyword_score = 0.0
        else:
            length_score = min(1.0, len(text) / 400.0)
            kw_hits = sum(1 for k in SCIENTIFIC_KEYWORDS if k in text)
            kw_score = min(1.0, kw_hits / 4.0)
            keyword_score = 0.6 * kw_score + 0.4 * length_score

        return float(0.5 * structural + 0.5 * keyword_score)
