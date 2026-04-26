"""Episode rollout helpers used by evaluate / infer (and by the GRPO trainer
for non-trainer-driven generation paths).

A rollout is one full env episode driven by a pluggable `generate_text`
callback. Every turn is logged via :class:`EpisodeLogger`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from drug_discovery_env.client import DrugDiscoveryClient
from drug_discovery_env.core.models import (
    STAGE_ORDER,
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
)
from drug_discovery_env.training.episode_logger import EpisodeLogger, TurnRecord
from drug_discovery_env.training.prompting import action_from_text


@dataclass
class EpisodeResult:
    episode_id: str
    disease: str
    steps: int
    terminal_reward: float
    total_reward: float
    reward_breakdown: Dict[str, float]
    final_stage: str
    final_stage_idx: int
    stage_completed: bool
    budget_remaining: float
    budget_total: float
    budget_remaining_frac: float
    oversight_violations: int
    nominated_smiles: Optional[str]
    nominated_compound_id: Optional[str]
    nominated_admet: Dict[str, Any]
    terminated_reason: str
    turn_outputs: List[str] = field(default_factory=list)
    turn_actions: List[DrugDiscoveryAction] = field(default_factory=list)


def run_episode(
    *,
    base_url: str,
    disease: str,
    generate_text: Callable[[DrugDiscoveryObservation], str],
    logger: EpisodeLogger,
    max_turns: int,
    episode_id: Optional[str] = None,
) -> EpisodeResult:
    eid = logger.start_episode(disease=disease, episode_id=episode_id)
    turn_outputs: List[str] = []
    turn_actions: List[DrugDiscoveryAction] = []
    breakdown: Dict[str, float] = {}
    last_obs: Optional[DrugDiscoveryObservation] = None
    final_reward = 0.0
    done = False
    terminated_reason = ""

    with DrugDiscoveryClient(base_url=base_url).sync() as env:
        step_result = env.reset(disease=disease)
        observation: DrugDiscoveryObservation = step_result.observation
        last_obs = observation
        turn = 0
        while not step_result.done and turn < max_turns:
            turn += 1
            raw_text = generate_text(observation)
            action = action_from_text(raw_text)
            turn_outputs.append(raw_text)
            turn_actions.append(action)

            step_result = env.step(action)
            observation = step_result.observation
            last_obs = observation
            done = bool(step_result.done)
            final_reward = float(step_result.reward or 0.0)

            tr_breakdown = (
                observation.reward_breakdown.model_dump()
                if observation.reward_breakdown is not None
                else {}
            )

            logger.log_turn(
                eid,
                TurnRecord(
                    step=observation.step_index,
                    stage=observation.stage,
                    tool=action.tool,
                    params=dict(action.params or {}),
                    target_compound_id=action.target_compound_id,
                    reasoning=action.reasoning,
                    evidence_ids=list(action.evidence_ids or []),
                    reward_breakdown=tr_breakdown,
                    budget_remaining=observation.budget_remaining,
                    budget_total=observation.budget_total,
                    last_tool_cost=observation.last_tool_cost,
                    done=done,
                    reward=final_reward,
                ),
            )

            if done:
                terminated_reason = str((observation.info or {}).get("terminated_reason") or "")
                breakdown = (
                    observation.reward_breakdown.model_dump()
                    if observation.reward_breakdown is not None
                    else {}
                )
                break

        if not done:
            terminated_reason = "max_turns_reached"

    nominated_smiles = None
    nominated_compound_id = None
    nominated_admet: Dict[str, Any] = {}
    final_stage = "target_selection"
    final_stage_idx = 0
    budget_remaining = 0.0
    budget_total = 0.0
    if last_obs is not None:
        final_stage = last_obs.stage
        final_stage_idx = STAGE_ORDER.index(last_obs.stage) if last_obs.stage in STAGE_ORDER else 0
        budget_remaining = float(last_obs.budget_remaining)
        budget_total = float(last_obs.budget_total or 0.0)
        nominated_compound_id = last_obs.advanced_compound_id
        for compound in last_obs.active_compounds or []:
            if compound.get("id") == nominated_compound_id:
                nominated_smiles = compound.get("smiles")
                nominated_admet = compound.get("admet") or {}
                break
        if nominated_smiles is None and last_obs.active_compounds:
            best = max(
                last_obs.active_compounds,
                key=lambda c: (
                    float(c.get("potency", 0.0))
                    + float(c.get("selectivity", 0.0))
                    + float(c.get("safety", 0.0))
                    + float(c.get("developability", 0.0))
                )
                / 4.0,
            )
            nominated_smiles = best.get("smiles")
            nominated_compound_id = best.get("id")
            nominated_admet = best.get("admet") or {}

    stage_completed = final_stage in {"finished", "lead_validation"} and bool(nominated_compound_id)
    budget_remaining_frac = budget_remaining / budget_total if budget_total > 0 else 0.0
    oversight_violations = 0
    if last_obs is not None:
        oversight_violations = int((last_obs.info or {}).get("oversight_violations", 0) or 0)

    terminal_reward = float(breakdown.get("terminal_compound", 0.0))
    total_reward = float(breakdown.get("total", final_reward))

    logger.end_episode(
        eid,
        disease=disease,
        steps=last_obs.step_index if last_obs else 0,
        terminal_reward=terminal_reward,
        total_reward=total_reward,
        stage_completed=stage_completed,
        budget_remaining_frac=budget_remaining_frac,
        oversight_violations=oversight_violations,
        nominated_compound_id=nominated_compound_id,
        nominated_smiles=nominated_smiles,
        nominated_admet=nominated_admet,
        terminated_reason=terminated_reason or "ok",
    )

    return EpisodeResult(
        episode_id=eid,
        disease=disease,
        steps=last_obs.step_index if last_obs else 0,
        terminal_reward=terminal_reward,
        total_reward=total_reward,
        reward_breakdown=breakdown,
        final_stage=final_stage,
        final_stage_idx=final_stage_idx,
        stage_completed=stage_completed,
        budget_remaining=budget_remaining,
        budget_total=budget_total,
        budget_remaining_frac=budget_remaining_frac,
        oversight_violations=oversight_violations,
        nominated_smiles=nominated_smiles,
        nominated_compound_id=nominated_compound_id,
        nominated_admet=nominated_admet,
        terminated_reason=terminated_reason or "ok",
        turn_outputs=turn_outputs,
        turn_actions=turn_actions,
    )
