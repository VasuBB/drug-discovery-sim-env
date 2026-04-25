"""Type-safe contracts for the Drug Discovery Sim Environment.

Wire format (Pydantic):
    Action       — what the Project Lead LLM decides each step
    Observation  — what it sees back
    State        — episode-level metadata
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from drug_discovery_env.openenv_compat import Action, Observation, State
except ModuleNotFoundError:  # pragma: no cover — fresh-bootstrap path
    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0


# Tool names the Project Lead can invoke. Validated server-side; kept as plain
# strings so the LLM can emit them naturally.
TOOL_NAMES = (
    "select_target",
    "search_compounds",
    "predict_affinity",
    "evaluate_admet",
    "modify_molecule",
    "synthesize",
    "validate_compound",
    "search_literature",
    "advance_stage",
    "abandon_compound",
    "pause_and_review_all",
    "delegate_to_subagent",
    "request_subagent_summary",
)


# Stage names — main's flow, used as strings throughout the env.
STAGE_ORDER = (
    "target_selection",
    "hit_id",
    "hit_to_lead",
    "admet",
    "lead_validation",
    "finished",
)


class DrugDiscoveryAction(Action):
    """A single decision from the Project Lead agent."""

    tool: str = Field(..., description="One of TOOL_NAMES")
    params: Dict[str, Any] = Field(default_factory=dict)
    target_compound_id: Optional[str] = Field(
        default=None,
        description="Which compound in the active pool this action applies to, if any",
    )
    reasoning: str = Field(default="", description="Free-form chain-of-thought")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Optional snippet IDs from prior literature searches the agent is grounding on",
    )


class ToolProvenance(BaseModel):
    source: str
    timestamp: str
    confidence: float


class RewardBreakdown(BaseModel):
    """All seven rubric components — sums to `total`."""

    terminal_compound: float = 0.0
    stage_progression: float = 0.0
    budget_efficiency: float = 0.0
    reasoning_depth: float = 0.0
    process: float = 0.0
    strategy: float = 0.0
    oversight_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class DrugDiscoveryObservation(Observation):
    """What the Project Lead sees after each action."""

    state_summary: str = Field(default="", description="Compact text summary of state")
    stage: str = Field(default="target_selection")
    step_index: int = 0
    max_steps: int = 50

    disease: str = ""
    selected_target: Optional[str] = None

    budget_total: float = 0.0
    budget_remaining: float = 0.0
    last_tool_cost: float = 0.0

    last_tool: Optional[str] = None
    last_result: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None  # akshat compatibility alias

    active_compounds: List[Dict[str, Any]] = Field(default_factory=list)
    advanced_compound_id: Optional[str] = None

    subagent_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{agent, severity: info|warn|block, message}, ...]",
    )

    provenance: Optional[ToolProvenance] = None
    uncertainty: Dict[str, float] = Field(default_factory=dict)
    reward_breakdown: Optional[RewardBreakdown] = None

    info: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="", description="Human-readable status line")


class DrugDiscoveryState(State):
    """Episode-level metadata."""

    disease: str = ""
    selected_target: Optional[str] = None
    max_steps: int = 50
    budget_total: float = 1000.0
    budget_remaining: float = 1000.0
    stage: str = "target_selection"
    compounds_tested: int = 0
    best_score: float = 0.0
    advanced_compound_id: Optional[str] = None
    terminated_reason: Optional[str] = None
