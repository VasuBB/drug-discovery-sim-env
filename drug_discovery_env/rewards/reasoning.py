"""Reasoning-depth reward — smooth multiplicative form.

    unique_concepts = |{kw in trace}|       (over a curated scientific lexicon)
    score           = tanh(unique / 6) * tanh(mean_len / 200)

The product keeps both axes (vocabulary breadth, sentence depth) honest:
neither alone can max the score.
"""

from __future__ import annotations

import math
import re
from typing import List

from drug_discovery_env.core.models import DrugDiscoveryAction

_CONCEPTS = (
    "binding", "affinity", "admet", "toxic", "pains", "lipinski",
    "scaffold", "selectivity", "potency", "docking", "novelty",
    "budget", "tradeoff", "trade-off", "hypothesis", "off-target",
    "modification", "candidate", "screen", "lead", "uncertainty",
    "evidence", "ro5", "qed", "logp", "hbd", "hba", "pIC50",
    "metabolism", "clearance", "selectivity", "efficacy",
)
_TRADEOFF_RE = re.compile(r"trade[- ]?off|balance|however")
_UNCERTAINTY_RE = re.compile(r"uncertain|confidence|risk")
_HYPOTHESIS_RE = re.compile(r"hypothes|expect|predict")


class ReasoningReward:
    def score(self, traces: List[str]) -> float:
        if not traces:
            return 0.0
        depths: List[float] = []
        lengths: List[float] = []
        for trace in traces:
            if not trace:
                depths.append(0.0)
                lengths.append(0.0)
                continue
            low = trace.lower()
            unique = len({k for k in _CONCEPTS if k in low})
            depths.append(math.tanh(unique / 6.0))
            lengths.append(math.tanh(len(trace) / 200.0))
        mean_depth = sum(depths) / len(depths)
        mean_length = sum(lengths) / len(lengths)
        return max(0.0, min(1.0, mean_depth * mean_length))

    def score_action(self, action: DrugDiscoveryAction) -> float:
        text = (action.reasoning or "").lower()
        bits = 0.0
        if _TRADEOFF_RE.search(text):
            bits += 0.25
        if _UNCERTAINTY_RE.search(text):
            bits += 0.25
        if action.evidence_ids:
            bits += 0.25
        if _HYPOTHESIS_RE.search(text):
            bits += 0.25
        return bits
