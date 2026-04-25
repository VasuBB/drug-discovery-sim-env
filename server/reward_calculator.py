"""Composable reward rubric for the drug discovery campaign.

Each rubric component is a small pure function returning a float in a known
range. The total reward is the sum. The components are deliberately coupled
to *different* dimensions of the agent's behaviour so that no single
component can be exploited in isolation:

    Component               Range          Targets
    ---------               -----          -------
    terminal compound        0..1.0        scientific quality of nominated lead
    stage progression        0..0.3        didn't skip critical checks
    budget efficiency        0..0.2        conserved budget
    reasoning depth          0..0.5        Mercor bonus: substantive reasoning
    oversight compliance    -0.2..0        penalty for ignoring sub-agent warnings
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RewardBreakdown:
    terminal_compound: float = 0.0
    stage_progression: float = 0.0
    budget_efficiency: float = 0.0
    reasoning_depth: float = 0.0
    oversight_penalty: float = 0.0

    def total(self) -> float:
        return (
            self.terminal_compound
            + self.stage_progression
            + self.budget_efficiency
            + self.reasoning_depth
            + self.oversight_penalty
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "terminal_compound": self.terminal_compound,
            "stage_progression": self.stage_progression,
            "budget_efficiency": self.budget_efficiency,
            "reasoning_depth": self.reasoning_depth,
            "oversight_penalty": self.oversight_penalty,
            "total": self.total(),
        }


def score_terminal_compound(
    advanced_compound: Optional[Dict],
    target: Optional[str],
    seen_smiles: List[str],
) -> float:
    if not advanced_compound:
        return 0.0
    affinity = advanced_compound.get("binding_affinity_nM")
    admet = advanced_compound.get("admet") or {}
    docking = advanced_compound.get("docking_score")

    score = 0.0
    if affinity is not None:
        score += 0.4 * max(0.0, 1.0 - math.log10(max(affinity, 0.1)) / 4.0)
    if docking is not None:
        clamped = max(-12.0, min(-4.0, docking))
        score += 0.3 * ((-clamped - 4.0) / 8.0)
    if admet:
        admet_term = 0.0
        if admet.get("ro5_pass"):
            admet_term += 0.5
        if not admet.get("pains", False):
            admet_term += 0.25
        if not admet.get("tox_flag", False):
            admet_term += 0.25
        score += 0.2 * admet_term
    novelty = 0.1
    if advanced_compound.get("smiles") in seen_smiles[:5]:
        novelty *= 0.2
    score += novelty
    return min(1.0, score)


def score_stage_progression(stages_completed: List[str]) -> float:
    canonical = ["target_selection", "hit_id", "hit_to_lead", "admet", "lead_validation"]
    cleared = 0
    for stage in canonical:
        if stage in stages_completed:
            cleared += 1
        else:
            break
    return 0.06 * cleared


def score_budget_efficiency(budget_remaining: float, budget_total: float) -> float:
    if budget_total <= 0:
        return 0.0
    return 0.2 * max(0.0, min(1.0, budget_remaining / budget_total))


def score_reasoning_depth(reasoning_traces: List[str]) -> float:
    if not reasoning_traces:
        return 0.0
    keywords = (
        "binding", "affinity", "admet", "toxic", "pains", "lipinski",
        "scaffold", "selectivity", "potency", "docking", "novelty",
        "budget", "trade-off", "tradeoff", "hypothesis", "off-target",
        "modification", "candidate", "screen", "lead",
    )
    per_step_scores = []
    for trace in reasoning_traces:
        if not trace:
            per_step_scores.append(0.0)
            continue
        length_score = min(1.0, len(trace) / 400.0)
        kw_hits = sum(1 for k in keywords if k in trace.lower())
        kw_score = min(1.0, kw_hits / 4.0)
        per_step_scores.append(0.6 * kw_score + 0.4 * length_score)
    avg = sum(per_step_scores) / len(per_step_scores)
    return 0.5 * avg


def score_oversight_compliance(warnings_issued: int, warnings_ignored: int) -> float:
    if warnings_issued == 0:
        return 0.0
    ignored_frac = warnings_ignored / warnings_issued
    return -0.2 * ignored_frac


def compute_reward(
    advanced_compound: Optional[Dict],
    target: Optional[str],
    seen_smiles: List[str],
    stages_completed: List[str],
    budget_remaining: float,
    budget_total: float,
    reasoning_traces: List[str],
    warnings_issued: int,
    warnings_ignored: int,
) -> RewardBreakdown:
    return RewardBreakdown(
        terminal_compound=score_terminal_compound(advanced_compound, target, seen_smiles),
        stage_progression=score_stage_progression(stages_completed),
        budget_efficiency=score_budget_efficiency(budget_remaining, budget_total),
        reasoning_depth=score_reasoning_depth(reasoning_traces),
        oversight_penalty=score_oversight_compliance(warnings_issued, warnings_ignored),
    )
