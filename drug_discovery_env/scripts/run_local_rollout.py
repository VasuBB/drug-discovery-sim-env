"""Run one in-process scripted rollout — quick smoke test."""

from __future__ import annotations

import argparse

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local rollout")
    parser.add_argument("--disease", type=str, required=True)
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.LIVE_ONLY
    env = DrugDiscoveryEnv(settings=settings)
    obs = env.reset(disease=args.disease)
    print("RESET:", obs.state_summary)

    scripted_actions = [
        f'<reasoning>Select target.</reasoning><tool>select_target</tool><params>{{"disease":"{args.disease}"}}</params>',
        "<reasoning>Need a candidate pool.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>",
    ]
    for action in scripted_actions:
        obs = env.step(action)
        print(obs.info, obs.reward_breakdown.total if obs.reward_breakdown else None)
        if obs.done:
            break

    if not obs.done:
        gs = getattr(env, "_game_state", None)
        if gs and gs.compound_ledger:
            smiles = next(iter(gs.compound_ledger.keys()))
            followups = [
                f'<reasoning>Quantify potency.</reasoning><tool>predict_affinity</tool><params>{{"smiles":"{smiles}","assay_type":"biochemical"}}</params>',
                f'<reasoning>Run ADMET.</reasoning><tool>evaluate_admet</tool><params>{{"smiles":"{smiles}"}}</params>',
                "<reasoning>Move to ADMET stage now.</reasoning><tool>advance_stage</tool><params>{}</params>",
                f'<reasoning>Final validation.</reasoning><tool>validate_compound</tool><params>{{"smiles":"{smiles}"}}</params>',
            ]
            for action in followups:
                obs = env.step(action)
                print(obs.info, obs.reward_breakdown.total if obs.reward_breakdown else None)
                if obs.done:
                    break

    print("FINAL:", obs.state_summary)


if __name__ == "__main__":
    main()
