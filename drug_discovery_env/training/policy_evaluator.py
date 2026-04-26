from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.model_policy import prompt_text_from_observation


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
    verbose: bool = False,
    debug_io: bool = False,
    episode_index: int = 1,
    total_episodes: int = 1,
) -> EpisodeSummary:
    observation = env.reset(disease=disease)
    done = False
    rewards: list[float] = []
    if verbose:
        print(
            f"[episode {episode_index}/{total_episodes}] start disease={disease} stage={observation.stage}",
            flush=True,
        )
    while not done:
        if debug_io:
            prompt = prompt_text_from_observation(observation)
            print(
                f"[episode {episode_index}/{total_episodes}] prompt:\n{prompt}\n",
                flush=True,
            )
        action = choose_action(observation)
        if debug_io:
            print(
                f"[episode {episode_index}/{total_episodes}] action={action}\n",
                flush=True,
            )
        observation = env.step(action)
        step_reward = observation.reward_breakdown.total if observation.reward_breakdown else float(observation.reward or 0.0)
        rewards.append(step_reward)
        if verbose:
            print(
                f"[episode {episode_index}/{total_episodes}] step={observation.step_index} "
                f"stage={observation.stage} tool={observation.last_tool} reward={step_reward:.4f} "
                f"done={observation.done}",
                flush=True,
            )
        if debug_io:
            print(
                f"[episode {episode_index}/{total_episodes}] result={observation.last_result}\n"
                f"[episode {episode_index}/{total_episodes}] message={observation.message}\n"
                f"[episode {episode_index}/{total_episodes}] next_state={observation.state_summary}\n",
                flush=True,
            )
        done = observation.done
    state = env.state
    if verbose:
        print(
            f"[episode {episode_index}/{total_episodes}] done total_reward={sum(rewards):.4f} "
            f"budget={state.budget_remaining:.2f} terminated={state.terminated_reason}",
            flush=True,
        )
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
    verbose: bool = False,
    debug_io: bool = False,
) -> list[EpisodeSummary]:
    return [
        run_episode(
            env,
            disease=disease,
            choose_action=choose_action,
            verbose=verbose,
            debug_io=debug_io,
            episode_index=index + 1,
            total_episodes=episodes,
        )
        for index in range(episodes)
    ]
