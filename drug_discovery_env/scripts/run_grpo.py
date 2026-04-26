from __future__ import annotations

import argparse
import json

from drug_discovery_env.config.settings import DataSourceMode
from drug_discovery_env.training.grpo_trainer import run_grpo_if_available


def _mode(value: str) -> DataSourceMode:
    return DataSourceMode(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO readiness or training")
    parser.add_argument("--train", action="store_true", help="Run actual GRPO trainer")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--episodes", type=int, default=3, help="Rollout episodes for dataset creation")
    parser.add_argument("--disease", type=str, default="Type 2 Diabetes")
    parser.add_argument(
        "--data-mode",
        type=_mode,
        choices=list(DataSourceMode),
        default=DataSourceMode.LOCAL_ONLY,
        help="Data source mode for rollout generation",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Training device selection",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--server-url", type=str, default=None, help="Run rollouts via running OpenEnv server")
    args = parser.parse_args()

    result = run_grpo_if_available(
        enable_actual_training=args.train,
        model_name_override=args.model,
        num_episodes=args.episodes,
        data_mode=args.data_mode,
        disease=args.disease,
        device=args.device,
        output_dir=args.output_dir,
        max_train_steps=args.max_train_steps,
        server_url=args.server_url,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
