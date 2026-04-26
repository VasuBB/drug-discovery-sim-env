"""Run GRPO readiness check or offline GRPO training.

For *live-rollout* GRPO against a running env, use train_grpo_live.py instead.
"""

from __future__ import annotations

import argparse
import json

from drug_discovery_env.config.settings import DataSourceMode
from drug_discovery_env.training.grpo_trainer import run_grpo_if_available
from drug_discovery_env.training.rollout_generator import normalize_diseases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO readiness or offline training")
    parser.add_argument("--train", action="store_true", help="Run actual GRPO trainer")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--episodes", type=int, default=3, help="Rollout episodes for dataset creation")
    parser.add_argument("--disease", type=str, default=None)
    parser.add_argument("--diseases", type=str, default=None, help="Comma-separated disease list for mixed training")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--debug-io", action="store_true", help="Print prompt, completion, env result, and reward during replay scoring")
    parser.add_argument("--debug-limit", type=int, default=10, help="Maximum number of replay samples to print when --debug-io is enabled")
    args = parser.parse_args()
    disease_list = normalize_diseases(
        disease=args.disease,
        diseases=args.diseases.split(",") if args.diseases else None,
    )
    if not disease_list:
        parser.error("Provide --disease or --diseases")

    result = run_grpo_if_available(
        enable_actual_training=args.train,
        model_name_override=args.model,
        num_episodes=args.episodes,
        data_mode=DataSourceMode.LIVE_ONLY,
        disease=disease_list[0],
        diseases=disease_list,
        device=args.device,
        output_dir=args.output_dir,
        max_train_steps=args.max_train_steps,
        debug_io=args.debug_io,
        debug_limit=args.debug_limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
