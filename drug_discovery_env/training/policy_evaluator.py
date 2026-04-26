from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.server.environment import DrugDiscoveryEnv


@dataclass
class EpisodeSummary:
    total_reward: float
    mean_step_reward: float
    final_budget: float
    final_stage_index: int
    terminated_reason: str | None


def scripted_action(observation: DrugDiscoveryObservation) -> str:
    stage = observation.stage
    actives = observation.active_compounds or []
    smiles = actives[0]["smiles"] if actives else None

    if stage == "target_selection":
        return (
            f'<reasoning>Select target.</reasoning><tool>select_target</tool>'
            f'<params>{{"disease":"{observation.disease}"}}</params>'
        )
    if stage == "hit_id" and not smiles:
        return "<reasoning>Search compounds.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>"
    if stage == "hit_id" and smiles:
        return (
            f'<reasoning>Affinity check.</reasoning><tool>predict_affinity</tool>'
            f'<params>{{"smiles":"{smiles}"}}</params>'
        )
    if stage == "hit_to_lead" and smiles:
        return (
            f'<reasoning>SAR variation.</reasoning><tool>modify_molecule</tool>'
            f'<params>{{"smiles":"{smiles}","instruction":"add fluorine"}}</params>'
        )
    if stage == "admet" and smiles:
        return f'<reasoning>ADMET.</reasoning><tool>evaluate_admet</tool><params>{{"smiles":"{smiles}"}}</params>'
    if stage == "lead_validation" and smiles:
        return (
            f'<reasoning>Validate.</reasoning><tool>validate_compound</tool>'
            f'<params>{{"smiles":"{smiles}"}}</params>'
        )
    return "<reasoning>Advance stage.</reasoning><tool>advance_stage</tool><params>{}</params>"


def run_episode(
    env: DrugDiscoveryEnv,
    *,
    disease: str,
    choose_action: Callable[[DrugDiscoveryObservation], str | DrugDiscoveryAction],
) -> EpisodeSummary:
    observation = env.reset(disease=disease)
    done = False
    rewards: list[float] = []
    while not done:
        action = choose_action(observation)
        observation = env.step(action)
        rewards.append(observation.reward_breakdown.total if observation.reward_breakdown else float(observation.reward or 0.0))
        done = observation.done
    state = env.state
    return EpisodeSummary(
        total_reward=sum(rewards),
        mean_step_reward=mean(rewards) if rewards else 0.0,
        final_budget=state.budget_remaining,
        final_stage_index=["target_selection", "hit_id", "hit_to_lead", "admet", "lead_validation", "finished"].index(state.stage),
        terminated_reason=state.terminated_reason,
    )


def evaluate_in_process(
    *,
    env: DrugDiscoveryEnv,
    disease: str,
    episodes: int,
    choose_action: Callable[[DrugDiscoveryObservation], str | DrugDiscoveryAction],
) -> list[EpisodeSummary]:
    return [run_episode(env, disease=disease, choose_action=choose_action) for _ in range(episodes)]
