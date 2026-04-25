from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    from drug_discovery_env.openenv_compat import Action, Observation, State
except ModuleNotFoundError:
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str | None = None
        step_count: int = 0


class DrugDiscoveryAction(Action):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    evidence_ids: list[str] = Field(default_factory=list)


class ToolProvenance(BaseModel):
    source: str
    timestamp: str
    confidence: float


class RewardBreakdown(BaseModel):
    terminal: float
    process: float
    reasoning: float
    strategy: float
    total: float


class DrugDiscoveryObservation(Observation):
    state_summary: str
    tool_result: dict[str, Any] | None = None
    provenance: ToolProvenance | None = None
    uncertainty: dict[str, float] = Field(default_factory=dict)
    sub_agent_messages: dict[str, list[str]] = Field(default_factory=dict)
    reward_breakdown: RewardBreakdown | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class DrugDiscoveryState(State):
    stage: int = 1
    budget_remaining: float = 0.0
    compounds_tested: int = 0
    best_score: float = 0.0
