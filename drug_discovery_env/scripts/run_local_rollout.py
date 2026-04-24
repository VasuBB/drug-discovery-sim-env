from __future__ import annotations

from drug_discovery_env.server.environment import DrugDiscoveryEnv


def main() -> None:
    env = DrugDiscoveryEnv()
    obs = env.reset("Type 2 Diabetes")
    print("RESET:", obs.state_summary)

    scripted_actions = [
        "<reasoning>Select best target for disease.</reasoning><tool>select_target</tool><params>{\"disease\":\"Type 2 Diabetes\"}</params>",
        "<reasoning>Need initial hit compounds before optimization.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>",
    ]

    for _ in range(2):
        for action in scripted_actions:
            obs = env.step(action)
            print(obs.info, obs.reward_breakdown.total if obs.reward_breakdown else None)
            if obs.done:
                break

    if not obs.done:
        any_smiles = next(iter(env.state.compound_ledger.keys()))
        followups = [
            f"<reasoning>Estimate potency with biochemical assay and known uncertainty.</reasoning><tool>predict_affinity</tool><params>{{\"smiles\":\"{any_smiles}\",\"assay_type\":\"biochemical\"}}</params>",
            f"<reasoning>Evaluate ADMET before costly validation.</reasoning><tool>evaluate_admet</tool><params>{{\"smiles\":\"{any_smiles}\"}}</params>",
            f"<reasoning>Ground next decisions in literature evidence.</reasoning><tool>search_literature</tool><params>{{\"query\":\"INSR PI3K safety selectivity\"}}</params><evidence>pmid_1001</evidence>",
            f"<reasoning>Validate with selective panel after safety checks.</reasoning><tool>validate_compound</tool><params>{{\"smiles\":\"{any_smiles}\"}}</params>",
        ]
        for action in followups:
            obs = env.step(action)
            print(obs.info, obs.reward_breakdown.total if obs.reward_breakdown else None)
            if obs.done:
                break

    print("FINAL:", obs.state_summary)


if __name__ == "__main__":
    main()
