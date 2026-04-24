from __future__ import annotations

from statistics import mean

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
        smiles = next(iter(env.state.compound_ledger.keys()), None) if env.state else None
        action = scripted_policy(env.state.step if env.state else 0, smiles)
        obs = env.step(action)
        rewards.append(obs.reward_breakdown.total if obs.reward_breakdown else 0.0)
        done = obs.done
    return {
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "final_budget": env.state.budget_remaining if env.state else 0.0,
        "final_stage": env.state.stage if env.state else 0,
    }


def main() -> None:
    env = DrugDiscoveryEnv()
    metrics = [run_episode(env) for _ in range(5)]
    print("Evaluation summary over 5 episodes")
    print("mean_total_reward", mean(m["total_reward"] for m in metrics))
    print("mean_step_reward", mean(m["mean_step_reward"] for m in metrics))
    print("mean_final_budget", mean(m["final_budget"] for m in metrics))
    print("mean_final_stage", mean(m["final_stage"] for m in metrics))


if __name__ == "__main__":
    main()
