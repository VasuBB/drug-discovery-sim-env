"""GameState: full episode state held inside the Environment.

Merges akshat-dev's multi-dim CompoundRecord/EvidenceRecord/ActionRecord with
main's per-compound history bookkeeping, oversight tracking, and named-stage
flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CompoundRecord:
    smiles: str
    id: Optional[str] = None
    origin: str = "unknown"  # "chembl" | "modified" | "synthesized" | "seed"
    parent_id: Optional[str] = None
    # multi-dim scores akshat used; populated by tools
    potency: float = 0.0
    selectivity: float = 0.0
    safety: float = 0.0
    synthesizability: float = 0.0
    novelty: float = 0.0
    developability: float = 0.0
    uncertainty: float = 1.0
    assay_count: int = 0
    # main-style fields used by reward + sub-agents
    binding_affinity_nM: Optional[float] = None
    docking_score: Optional[float] = None
    admet: Optional[Dict[str, Any]] = None
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    step: int
    tool: str
    params: Dict[str, Any]
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
    stage: str = "target_selection"
    step: int = 0
    max_steps: int = 50

    budget_initial: float = 1000.0
    budget_remaining: float = 1000.0

    target: Optional[Dict[str, Any]] = None
    selected_target: Optional[str] = None
    target_class: Optional[str] = None

    pathway_graph: Dict[str, List[str]] = field(default_factory=dict)
    disease_nodes: List[str] = field(default_factory=list)
    off_target_risk_profile: Dict[str, float] = field(default_factory=dict)
    assay_confidence: Dict[str, float] = field(default_factory=dict)
    uncertainty_estimates: Dict[str, float] = field(default_factory=dict)
    experiment_queue: List[Dict[str, Any]] = field(default_factory=list)

    # compound store (keyed by SMILES — lookup compatibility with akshat tools)
    compound_ledger: Dict[str, CompoundRecord] = field(default_factory=dict)
    # parallel index by compound_id (e.g. "C001") for main-style references
    compound_id_index: Dict[str, str] = field(default_factory=dict)
    _next_compound_idx: int = 0

    best_candidate_frontier: List[str] = field(default_factory=list)
    evidence_ledger: Dict[str, EvidenceRecord] = field(default_factory=dict)
    assay_history: List[Dict[str, Any]] = field(default_factory=list)
    budget_ledger: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[ActionRecord] = field(default_factory=list)

    sub_agent_inbox: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    # oversight bookkeeping (main's pattern)
    reasoning_traces: List[str] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    warnings_issued: int = 0
    warnings_ignored: int = 0
    open_block_warnings: List[Dict[str, str]] = field(default_factory=list)

    advanced_compound_id: Optional[str] = None
    terminated_reason: Optional[str] = None

    last_tool: Optional[str] = None
    last_result: Dict[str, Any] = field(default_factory=dict)
    last_cost: float = 0.0
    last_message: str = ""

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

    def best_compound(self) -> Optional[CompoundRecord]:
        if not self.compound_ledger:
            return None
        return max(
            self.compound_ledger.values(),
            key=lambda c: (c.potency + c.selectivity + c.safety + c.developability) / 4,
        )

    def new_compound_id(self) -> str:
        self._next_compound_idx += 1
        return f"C{self._next_compound_idx:03d}"

    def add_compound(
        self,
        smiles: str,
        *,
        origin: str = "chembl",
        parent_id: Optional[str] = None,
        history: Optional[str] = None,
        **fields: Any,
    ) -> CompoundRecord:
        if smiles in self.compound_ledger:
            rec = self.compound_ledger[smiles]
            if history:
                rec.history.append(history)
            for k, v in fields.items():
                if v is not None and hasattr(rec, k):
                    setattr(rec, k, v)
            return rec
        cid = self.new_compound_id()
        rec = CompoundRecord(
            smiles=smiles,
            id=cid,
            origin=origin,
            parent_id=parent_id,
            history=[history] if history else [],
        )
        for k, v in fields.items():
            if v is not None and hasattr(rec, k):
                setattr(rec, k, v)
        self.compound_ledger[smiles] = rec
        self.compound_id_index[cid] = smiles
        return rec

    def get_by_id(self, compound_id: str) -> Optional[CompoundRecord]:
        smiles = self.compound_id_index.get(compound_id)
        if not smiles:
            return None
        return self.compound_ledger.get(smiles)

    def active_compounds(self) -> List[CompoundRecord]:
        return list(self.compound_ledger.values())
