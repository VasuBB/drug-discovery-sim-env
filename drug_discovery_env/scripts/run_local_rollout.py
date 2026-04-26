from __future__ import annotations

import argparse
import json
import re

from drug_discovery_env.client import create_sync_client
from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction


def _mode(value: str) -> DataSourceMode:
    return DataSourceMode(value)


def _to_action(raw: str) -> DrugDiscoveryAction:
    tool = re.search(r"<tool>(.*?)</tool>", raw, re.DOTALL)
    params = re.search(r"<params>(.*?)</params>", raw, re.DOTALL)
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    return DrugDiscoveryAction(
        tool=tool.group(1).strip() if tool else "search_compounds",
        params=json.loads(params.group(1).strip()) if params else {},
        reasoning=reasoning.group(1).strip() if reasoning else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local rollout")
    parser.add_argument("--disease", type=str, default="Type 2 Diabetes")
    parser.add_argument(
        "--data-mode",
        type=_mode,
        choices=list(DataSourceMode),
        default=DataSourceMode.LOCAL_ONLY,
        help="Data source mode",
    )
    parser.add_argument("--server-url", type=str, default=None, help="Use running server via client")
    args = parser.parse_args()

    scripted_actions = [
        "<reasoning>Select best target for disease.</reasoning><tool>select_target</tool><params>{\"disease\":\"Type 2 Diabetes\"}</params>",
        "<reasoning>Need initial hit compounds before optimization.</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>",
    ]

    if args.server_url:
        with create_sync_client(args.server_url) as client:
            obs = client.reset(disease=args.disease).observation
            print("RESET:", obs.state_summary)

            done = False
            last_smiles: str | None = None
            for _ in range(2):
                for action in scripted_actions:
                    step_result = client.step(_to_action(action))
                    obs = step_result.observation
                    print(obs.info, step_result.reward)
                    hits = (obs.tool_result or {}).get("hits", []) if isinstance(obs.tool_result, dict) else []
                    if hits and isinstance(hits[0], dict) and hits[0].get("smiles"):
                        last_smiles = str(hits[0]["smiles"])
                    if step_result.done:
                        done = True
                        break
                if done:
                    break

            if not done and last_smiles:
                followups = [
                    f"<reasoning>Estimate potency with biochemical assay and known uncertainty.</reasoning><tool>predict_affinity</tool><params>{{\"smiles\":\"{last_smiles}\",\"assay_type\":\"biochemical\"}}</params>",
                    f"<reasoning>Evaluate ADMET before costly validation.</reasoning><tool>evaluate_admet</tool><params>{{\"smiles\":\"{last_smiles}\"}}</params>",
                    f"<reasoning>Ground next decisions in literature evidence.</reasoning><tool>search_literature</tool><params>{{\"query\":\"INSR PI3K safety selectivity\"}}</params><evidence>pmid_1001</evidence>",
                    f"<reasoning>Validate with selective panel after safety checks.</reasoning><tool>validate_compound</tool><params>{{\"smiles\":\"{last_smiles}\"}}</params>",
                ]
                for action in followups:
                    step_result = client.step(_to_action(action))
                    obs = step_result.observation
                    print(obs.info, step_result.reward)
                    if step_result.done:
                        break

            print("FINAL:", obs.state_summary)
        return

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = args.data_mode
    from drug_discovery_env.server.environment import DrugDiscoveryEnv

    env = DrugDiscoveryEnv(settings=settings)
    obs = env.reset(disease=args.disease)
    print("RESET:", obs.state_summary)

    for _ in range(2):
        for action in scripted_actions:
            obs = env.step(action)
            print(obs.info, obs.reward_breakdown.total if obs.reward_breakdown else None)
            if obs.done:
                break

    if not obs.done:
        game_state = getattr(env, "_game_state", None)
        if game_state is None or not game_state.compound_ledger:
            print("FINAL:", obs.state_summary)
            return
        any_smiles = next(iter(game_state.compound_ledger.keys()))
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
