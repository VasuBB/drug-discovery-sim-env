"""Reasoning-depth reward — continuous keyword + length scoring per trace.

Mercor bonus: reward substantive scientific reasoning per step. Combines
domain-keyword density (60%) with trace length (40%). Also gives partial
credit for evidence-grounded actions and tradeoff/uncertainty markers.
"""

from __future__ import annotations

import re
from typing import List

from drug_discovery_env.core.models import DrugDiscoveryAction


_KEYWORDS = (
    "binding", "affinity", "admet", "toxic", "pains", "lipinski",
    "scaffold", "selectivity", "potency", "docking", "novelty",
    "budget", "trade-off", "tradeoff", "hypothesis", "off-target",
    "modification", "candidate", "screen", "lead", "uncertainty",
    "evidence", "ro5", "qed", "logp",
)
_TRADEOFF_RE = re.compile(r"trade[- ]?off|balance|however")
_UNCERTAINTY_RE = re.compile(r"uncertain|confidence|risk")
_HYPOTHESIS_RE = re.compile(r"hypothesis|expect|predict")


class ReasoningReward:
    def score(self, traces: List[str]) -> float:
        if not traces:
            return 0.0
        per_step: List[float] = []
        for trace in traces:
            if not trace:
                per_step.append(0.0)
                continue
            length_score = min(1.0, len(trace) / 400.0)
            low = trace.lower()
            kw_hits = sum(1 for k in _KEYWORDS if k in low)
            kw_score = min(1.0, kw_hits / 4.0)
            per_step.append(0.6 * kw_score + 0.4 * length_score)
        return sum(per_step) / len(per_step)

    def score_action(self, action: DrugDiscoveryAction) -> float:
        """Per-action shaping bonus used during dense process reward."""
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
