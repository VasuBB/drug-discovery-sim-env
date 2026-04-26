"""Generate offline rollouts used to seed GRPO training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.model_policy import prompt_text_from_observation


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    reward: float
    disease: str
    history: list[str]


def _json_action(
    tool: str,
    params: dict[str, object],
    reasoning: str,
    *,
    target_compound_id: str | None = None,
) -> str:
    return json.dumps(
        {
            "tool": tool,
            "params": params,
            "target_compound_id": target_compound_id,
            "reasoning": reasoning,
        }
    )


def heuristic_policy(state_summary: str, candidate_smiles: Optional[str] = None) -> str:
    """Stage-aware scripted policy used to bootstrap offline rollouts."""
    if "stage=target_selection" in state_summary:
        return _json_action(
            "select_target",
            {},
            "Select the strongest disease target before any chemistry work.",
        )
    if "compounds=0" in state_summary:
        return _json_action(
            "search_compounds",
            {"min_qed": 0.5},
            "Need a live candidate pool before any optimization.",
        )
    if candidate_smiles:
        return _json_action(
            "predict_affinity",
            {"smiles": candidate_smiles, "assay_type": "biochemical"},
            "Measure potency before expensive validation to reduce uncertainty efficiently.",
        )
    return _json_action(
        "search_literature",
        {"query": "target safety selectivity"},
        "Ground the next decision in literature evidence to reduce uncertainty.",
    )


def generate_rollouts(
    num_episodes: int = 3,
    *,
    settings: Optional[Settings] = None,
    data_mode: Optional[DataSourceMode] = None,
    disease: str,
) -> list[RolloutSample]:
    cfg = settings.model_copy(deep=True) if settings else get_settings().model_copy(deep=True)
    if data_mode is not None:
        if data_mode != DataSourceMode.LIVE_ONLY:
            raise ValueError("Only live_only data mode is supported")
        cfg.data.mode = data_mode
    if not disease:
        raise ValueError("disease is required for rollout generation")

    env = DrugDiscoveryEnv(settings=cfg)
    samples: list[RolloutSample] = []
    for _ in range(num_episodes):
        obs = env.reset(disease=disease)
        done = False
        history: list[str] = []
        while not done:
            prompt = prompt_text_from_observation(obs)
            game_state = getattr(env, "_game_state", None)
            candidate = next(iter(game_state.compound_ledger.keys()), None) if game_state else None
            completion = heuristic_policy(obs.state_summary, candidate_smiles=candidate)
            obs = env.step(completion)
            reward = obs.reward_breakdown.total if obs.reward_breakdown else float(obs.reward or 0.0)
            samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=completion,
                    reward=reward,
                    disease=disease,
                    history=list(history),
                )
            )
            history.append(completion)
            done = obs.done
    return samples
