from __future__ import annotations

from dataclasses import dataclass

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    reward: float


def heuristic_policy(state_summary: str, candidate_smiles: str | None = None) -> str:
    if "stage=1" in state_summary:
        return "<reasoning>Select strongest disease target first.</reasoning><tool>select_target</tool><params>{}</params>"
    if "compounds=0" in state_summary:
        return "<reasoning>Need a candidate pool before optimization.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>"
    if candidate_smiles:
        return (
            "<reasoning>Measure potency before expensive validation.</reasoning>"
            "<tool>predict_affinity</tool>"
            f"<params>{{\"smiles\":\"{candidate_smiles}\",\"assay_type\":\"biochemical\"}}</params>"
        )
    return "<reasoning>Ground decision with literature first.</reasoning><tool>search_literature</tool><params>{\"query\":\"diabetes target safety\"}</params>"


def generate_rollouts(
    num_episodes: int = 3,
    *,
    settings: Settings | None = None,
    data_mode: DataSourceMode | None = None,
    disease: str = "Type 2 Diabetes",
) -> list[RolloutSample]:
    cfg = settings.model_copy(deep=True) if settings else get_settings().model_copy(deep=True)
    if data_mode is not None:
        cfg.data.mode = data_mode
    env = DrugDiscoveryEnv(settings=cfg)
    samples: list[RolloutSample] = []
    for _ in range(num_episodes):
        obs = env.reset(disease=disease)
        done = False
        while not done:
            prompt = obs.state_summary
            game_state = getattr(env, "_game_state", None)
            candidate = next(iter(game_state.compound_ledger.keys()), None) if game_state else None
            completion = heuristic_policy(prompt, candidate_smiles=candidate)
            obs = env.step(completion)
            reward = obs.reward_breakdown.total if obs.reward_breakdown else 0.0
            samples.append(RolloutSample(prompt=prompt, completion=completion, reward=reward))
            done = obs.done
    return samples
