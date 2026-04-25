from __future__ import annotations

import argparse
from statistics import mean

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv


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


def run_episode(env: DrugDiscoveryEnv) -> dict[str, float]:
    obs = env.reset()
    done = False
    rewards: list[float] = []
    while not done:
        game_state = getattr(env, "_game_state", None)
        smiles = next(iter(game_state.compound_ledger.keys()), None) if game_state else None
        step_idx = game_state.step if game_state else 0
        action = scripted_policy(step_idx, smiles)
        obs = env.step(action)
        rewards.append(obs.reward_breakdown.total if obs.reward_breakdown else 0.0)
        done = obs.done
    return {
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "final_budget": game_state.budget_remaining if game_state else 0.0,
        "final_stage": game_state.stage if game_state else 0,
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
        default=DataSourceMode.HYBRID,
        help="Data source mode",
    )
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = args.data_mode
    env = DrugDiscoveryEnv(settings=settings)
    metrics = [run_episode(env) for _ in range(args.episodes)]
    print(f"Evaluation summary over {args.episodes} episodes")
    print("mean_total_reward", mean(m["total_reward"] for m in metrics))
    print("mean_step_reward", mean(m["mean_step_reward"] for m in metrics))
    print("mean_final_budget", mean(m["final_budget"] for m in metrics))
    print("mean_final_stage", mean(m["final_stage"] for m in metrics))


if __name__ == "__main__":
    main()
