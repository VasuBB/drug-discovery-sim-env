from __future__ import annotations

import argparse
import json
import re
from statistics import mean
from typing import Any

from drug_discovery_env.client import create_sync_client
from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction


def scripted_policy(step: int, smiles: str | None) -> str:
    if step == 0:
        return "<reasoning>Select high-confidence target.</reasoning><tool>select_target</tool><params>{}</params>"
    if step == 1:
        return "<reasoning>Search compounds before deep assays.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>"
    if step == 2 and smiles:
        return f"<reasoning>Quantify affinity with controlled assay noise.</reasoning><tool>predict_affinity</tool><params>{{\"smiles\":\"{smiles}\",\"assay_type\":\"biochemical\"}}</params>"
    if step == 3 and smiles:
        return f"<reasoning>Run ADMET early to reduce late-stage failure risk.</reasoning><tool>evaluate_admet</tool><params>{{\"smiles\":\"{smiles}\"}}</params>"
    if smiles:
        return f"<reasoning>Validate candidate with selective panel.</reasoning><tool>validate_compound</tool><params>{{\"smiles\":\"{smiles}\"}}</params>"
    return "<reasoning>Use literature to ground decision.</reasoning><tool>search_literature</tool><params>{\"query\":\"disease target selectivity safety\"}</params>"


def _to_action(raw: str) -> DrugDiscoveryAction:
    tool = re.search(r"<tool>(.*?)</tool>", raw, re.DOTALL)
    params = re.search(r"<params>(.*?)</params>", raw, re.DOTALL)
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    return DrugDiscoveryAction(
        tool=tool.group(1).strip() if tool else "search_compounds",
        params=json.loads(params.group(1).strip()) if params else {},
        reasoning=reasoning.group(1).strip() if reasoning else "",
    )


def run_episode_local(env: Any) -> dict[str, float]:
    obs = env.reset()
    done = False
    rewards: list[float] = []
    while not done:
        game_state = getattr(env, "_game_state", None)
        smiles = next(iter(game_state.compound_ledger.keys()), None) if game_state else None
        step_idx = game_state.step if game_state else 0
        action = scripted_policy(step_idx, smiles)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        done = obs.done
    return {
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "final_budget": game_state.budget_remaining if game_state else 0.0,
        "final_stage": game_state.stage if game_state else 0,
    }


def run_episode_remote(server_url: str) -> dict[str, float]:
    with create_sync_client(server_url) as client:
        obs = client.reset().observation
        done = False
        rewards: list[float] = []
        step_idx = 0
        smiles: str | None = None
        final_budget = 0.0
        final_stage = 0
        while not done:
            action = scripted_policy(step_idx, smiles)
            step = client.step(_to_action(action))
            obs = step.observation
            rewards.append(float(step.reward if step.reward is not None else (obs.reward or 0.0)))
            hits = (obs.tool_result or {}).get("hits", []) if isinstance(obs.tool_result, dict) else []
            if hits and isinstance(hits[0], dict) and hits[0].get("smiles"):
                smiles = str(hits[0]["smiles"])
            if isinstance(obs.tool_result, dict) and isinstance(obs.tool_result.get("smiles"), str):
                smiles = str(obs.tool_result["smiles"])
            step_idx += 1
            done = bool(step.done)
            st = client.state()
            final_budget = float(getattr(st, "budget_remaining", 0.0))
            final_stage = int(getattr(st, "stage", 0))

    return {
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "final_budget": final_budget,
        "final_stage": final_stage,
    }


def _mode(value: str) -> DataSourceMode:
    return DataSourceMode(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation rollouts")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--data-mode",
        type=_mode,
        choices=list(DataSourceMode),
        default=DataSourceMode.LOCAL_ONLY,
        help="Data source mode",
    )
    parser.add_argument("--server-url", type=str, default=None, help="Use running server via client")
    args = parser.parse_args()

    if args.server_url:
        metrics = [run_episode_remote(args.server_url) for _ in range(args.episodes)]
    else:
        from drug_discovery_env.server.environment import DrugDiscoveryEnv

        settings = get_settings().model_copy(deep=True)
        settings.data.mode = args.data_mode
        env = DrugDiscoveryEnv(settings=settings)
        metrics = [run_episode_local(env) for _ in range(args.episodes)]

    print(f"Evaluation summary over {args.episodes} episodes")
    print("mean_total_reward", mean(m["total_reward"] for m in metrics))
    print("mean_step_reward", mean(m["mean_step_reward"] for m in metrics))
    print("mean_final_budget", mean(m["final_budget"] for m in metrics))
    print("mean_final_stage", mean(m["final_stage"] for m in metrics))


if __name__ == "__main__":
    main()
