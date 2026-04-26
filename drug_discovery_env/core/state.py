from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CompoundRecord:
    smiles: str
    potency: float = 0.0
    selectivity: float = 0.0
    safety: float = 0.0
    synthesizability: float = 0.0
    novelty: float = 0.0
    developability: float = 0.0
    uncertainty: float = 1.0
    assay_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    step: int
    tool: str
    params: dict[str, Any]
    information_gain: float
    cost_paid: float


@dataclass
class EvidenceRecord:
    evidence_id: str
    title: str
    snippet: str
    score: float
    source: str
    timestamp: str
    confidence: float


@dataclass
class GameState:
    disease: str
    stage: int = 1
    step: int = 0
    budget_initial: float = 500.0
    budget_remaining: float = 500.0
    target: dict[str, Any] | None = None
    target_class: str | None = None
    pathway_graph: dict[str, list[str]] = field(default_factory=dict)
    disease_nodes: list[str] = field(default_factory=list)
    off_target_risk_profile: dict[str, float] = field(default_factory=dict)
    assay_confidence: dict[str, float] = field(default_factory=dict)
    uncertainty_estimates: dict[str, float] = field(default_factory=dict)
    experiment_queue: list[dict[str, Any]] = field(default_factory=list)
    compound_ledger: dict[str, CompoundRecord] = field(default_factory=dict)
    best_candidate_frontier: list[str] = field(default_factory=list)
    evidence_ledger: dict[str, EvidenceRecord] = field(default_factory=dict)
    assay_history: list[dict[str, Any]] = field(default_factory=list)
    budget_ledger: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[ActionRecord] = field(default_factory=list)
    sub_agent_inbox: dict[str, list[str]] = field(default_factory=dict)
    last_potential_score: float = 0.0

    def add_budget_event(self, reason: str, amount: float) -> None:
        self.budget_remaining = max(0.0, self.budget_remaining - amount)
        self.budget_ledger.append(
            {
                "step": self.step,
                "reason": reason,
                "amount": amount,
                "remaining": self.budget_remaining,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def best_compound(self) -> CompoundRecord | None:
        if not self.compound_ledger:
            return None
        return max(
            self.compound_ledger.values(),
            key=lambda c: (c.potency + c.selectivity + c.safety + c.developability) / 4,
        )
