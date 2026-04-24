from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DrugDiscoveryAction(BaseModel):
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


class DrugDiscoveryObservation(BaseModel):
    state_summary: str
    tool_result: dict[str, Any] | None = None
    provenance: ToolProvenance | None = None
    uncertainty: dict[str, float] = Field(default_factory=dict)
    sub_agent_messages: dict[str, list[str]] = Field(default_factory=dict)
    reward_breakdown: RewardBreakdown | None = None
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
