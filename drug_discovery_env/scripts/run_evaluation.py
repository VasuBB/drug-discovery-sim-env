"""Run evaluation rollouts with a scripted policy, report mean rewards."""

from __future__ import annotations

import argparse
from statistics import mean

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv


def scripted_action(stage: str, smiles: str | None) -> str:
    if stage == "target_selection":
        return "<reasoning>Select target.</reasoning><tool>select_target</tool><params>{}</params>"
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


def run_episode(env: DrugDiscoveryEnv) -> dict[str, float]:
    obs = env.reset()
    done = False
    rewards: list[float] = []
    while not done:
        gs = getattr(env, "_game_state", None)
        smiles = next(iter(gs.compound_ledger.keys()), None) if gs else None
        action = scripted_action(obs.stage, smiles)
        obs = env.step(action)
        rewards.append(obs.reward_breakdown.total if obs.reward_breakdown else float(obs.reward or 0.0))
        done = obs.done
    state = env.state
    return {
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "final_budget": state.budget_remaining,
        "final_stage_index": ["target_selection", "hit_id", "hit_to_lead", "admet", "lead_validation", "finished"].index(state.stage),
    }


def _mode(value: str) -> DataSourceMode:
    return DataSourceMode(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation rollouts")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--data-mode", type=_mode, choices=list(DataSourceMode), default=DataSourceMode.HYBRID)
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = args.data_mode
    env = DrugDiscoveryEnv(settings=settings)
    metrics = [run_episode(env) for _ in range(args.episodes)]
    print(f"Evaluation summary over {args.episodes} episodes")
    print("mean_total_reward", mean(m["total_reward"] for m in metrics))
    print("mean_step_reward", mean(m["mean_step_reward"] for m in metrics))
    print("mean_final_budget", mean(m["final_budget"] for m in metrics))
    print("mean_final_stage_index", mean(m["final_stage_index"] for m in metrics))


if __name__ == "__main__":
    main()
