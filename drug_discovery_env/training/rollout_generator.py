from __future__ import annotations

from dataclasses import dataclass

from drug_discovery_env.client import create_sync_client
from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings


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
    server_url: str | None = None,
) -> list[RolloutSample]:
    cfg = settings.model_copy(deep=True) if settings else get_settings().model_copy(deep=True)
    if data_mode is not None:
        cfg.data.mode = data_mode

    def _extract_candidate(obs: DrugDiscoveryObservation, fallback: str | None) -> str | None:
        result = obs.tool_result or {}
        if isinstance(result.get("hits"), list) and result["hits"]:
            first = result["hits"][0]
            if isinstance(first, dict) and first.get("smiles"):
                return str(first["smiles"])
        if isinstance(result.get("smiles"), str):
            return result["smiles"]
        return fallback

    samples: list[RolloutSample] = []
    if server_url:
        with create_sync_client(server_url) as client:
            for _ in range(num_episodes):
                obs = client.reset(disease=disease).observation
                done = False
                candidate_smiles: str | None = None
                while not done:
                    prompt = obs.state_summary
                    completion = heuristic_policy(prompt, candidate_smiles=candidate_smiles)
                    action = DrugDiscoveryAction(tool="search_compounds", reasoning="fallback", params={})
                    if "<tool>" in completion:
                        # Reuse env parser semantics by passing raw text via client only if raw mode supported.
                        # Here we parse minimally for known tags.
                        import json, re

                        tool_match = re.search(r"<tool>(.*?)</tool>", completion, re.DOTALL)
                        params_match = re.search(r"<params>(.*?)</params>", completion, re.DOTALL)
                        reason_match = re.search(r"<reasoning>(.*?)</reasoning>", completion, re.DOTALL)
                        tool = tool_match.group(1).strip() if tool_match else "search_compounds"
                        params = json.loads(params_match.group(1).strip()) if params_match else {}
                        reasoning = reason_match.group(1).strip() if reason_match else ""
                        action = DrugDiscoveryAction(tool=tool, params=params, reasoning=reasoning)

                    step_result = client.step(action)
                    obs = step_result.observation
                    candidate_smiles = _extract_candidate(obs, candidate_smiles)
                    reward = float(step_result.reward if step_result.reward is not None else (obs.reward or 0.0))
                    samples.append(RolloutSample(prompt=prompt, completion=completion, reward=reward))
                    done = bool(step_result.done)
    else:
        from drug_discovery_env.server.environment import DrugDiscoveryEnv

        env = DrugDiscoveryEnv(settings=cfg)
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
