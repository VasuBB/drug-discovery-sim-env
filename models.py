"""Type-safe contracts for the Drug Discovery Sim Environment.

These Pydantic models define the wire format between client and server.
Action: what the Project Lead LLM decides each step.
Observation: what it sees back (current stage, compounds, sub-agent messages, budget).
State: episode-level metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

# Tool names the Project Lead can invoke. Kept as plain strings (not an Enum)
# so the LLM can emit them naturally and we can validate server-side.
TOOL_NAMES = (
    "search_chembl",
    "predict_binding_affinity",
    "compute_admet",
    "modify_molecule",
    "run_docking",
    "literature_search",
    "delegate_to_subagent",
    "advance_stage",
    "abandon_compound",
    "pause_and_review_all",
    "request_subagent_summary",
)


class DrugDiscoveryAction(Action):
    """A single decision from the Project Lead agent."""

    tool: str = Field(..., description="One of TOOL_NAMES")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific arguments (disease, target, smiles, instruction, query, subagent, ...)",
    )
    target_compound_id: Optional[str] = Field(
        default=None,
        description="Which compound in the active pool this action applies to, if any",
    )
    reasoning: str = Field(
        default="",
        description="Free-form chain-of-thought; scored by the reasoning-depth rubric",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class CompoundRecord(Observation.__base__ if False else object):  # plain dataclass-like model below
    pass


# We model nested structures as plain dicts in the observation payload to keep
# the wire format simple and JSON-friendly. The schemas below document them.

#   Compound dict: {
#       "id": str,
#       "smiles": str,
#       "origin": "chembl" | "modified" | "seed",
#       "binding_affinity_nM": Optional[float],
#       "admet": Optional[Dict[str, Any]],   # {"ro5_pass": bool, "pains": bool, "tox_score": float, "logp": float, ...}
#       "docking_score": Optional[float],
#       "history": List[str],                # human-readable log of what was done to it
#   }
#
#   SubAgentMessage dict: {
#       "agent": "chemist" | "toxicologist" | "budget" | "oversight",
#       "severity": "info" | "warn" | "block",
#       "message": str,
#   }


class DrugDiscoveryObservation(Observation):
    """What the Project Lead sees after each action.

    `done` and `reward` are inherited from the OpenEnv base.
    """

    stage: str = Field(..., description="target_selection | hit_id | hit_to_lead | admet | lead_validation | finished")
    step_index: int = Field(..., description="0-based step counter within the campaign")
    max_steps: int = Field(..., description="Hard cap on steps for this campaign")

    disease: str = Field(..., description="The disease context for this campaign")
    selected_target: Optional[str] = Field(default=None, description="Protein target chosen in stage 1")

    budget_total: float = Field(..., description="Starting budget (arbitrary units)")
    budget_remaining: float = Field(..., description="Budget left at this point")
    last_tool_cost: float = Field(default=0.0, description="What the most recent action cost")

    last_tool: Optional[str] = Field(default=None, description="Name of the last tool invoked")
    last_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload returned by the last tool call",
    )

    active_compounds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Compounds currently being tracked in this campaign",
    )
    advanced_compound_id: Optional[str] = Field(
        default=None,
        description="The lead candidate the agent has nominated, if any",
    )

    subagent_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Messages emitted by Chemist / Toxicologist / Budget / Oversight this step",
    )

    message: str = Field(default="", description="Human-readable status line for the agent")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DrugDiscoveryState(State):
    """Episode-level metadata. Inherits episode_id and step_count."""

    disease: str = ""
    selected_target: Optional[str] = None
    max_steps: int = 50
    budget_total: float = 1000.0
    budget_remaining: float = 1000.0
    stage: str = "target_selection"
    advanced_compound_id: Optional[str] = None
    terminated_reason: Optional[str] = None  # "completed" | "budget_exhausted" | "max_steps" | "abandoned"
