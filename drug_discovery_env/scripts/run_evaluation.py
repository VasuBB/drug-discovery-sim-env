"""Run evaluation rollouts with a scripted or model-based policy."""

from __future__ import annotations

import argparse
from statistics import mean

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.model_policy import TransformersToolPolicy
from drug_discovery_env.training.policy_evaluator import evaluate_in_process, scripted_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation rollouts")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--policy", choices=["scripted", "model"], default="scripted")
    parser.add_argument("--model", type=str, default=None, help="Model name or checkpoint path for --policy model")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true", help="Print per-episode and per-step progress")
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.LIVE_ONLY
    env = DrugDiscoveryEnv(settings=settings)

    model_ref = None
    if args.policy == "scripted":
        chooser = scripted_action
    else:
        if not args.model:
            parser.error("--model is required when --policy model")
        model_ref = args.model
        policy = TransformersToolPolicy(
            args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        chooser = policy.next_action

    metrics = evaluate_in_process(
        env=env,
        disease=args.disease,
        episodes=args.episodes,
        choose_action=chooser,
        verbose=args.verbose,
    )
    print(f"Evaluation summary over {args.episodes} episodes")
    print("policy", args.policy)
    if model_ref:
        print("model", model_ref)
    print("mean_total_reward", mean(m.total_reward for m in metrics))
    print("mean_step_reward", mean(m.mean_step_reward for m in metrics))
    print("mean_final_budget", mean(m.final_budget for m in metrics))
    print("mean_final_stage_index", mean(m.final_stage_index for m in metrics))


if __name__ == "__main__":
    main()
