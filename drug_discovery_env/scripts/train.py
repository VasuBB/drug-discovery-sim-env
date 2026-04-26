"""GRPO training entrypoint — server-client live rollouts only.

Talks to a running env server over HTTP (`training.base_url`), samples
diseases from the cached train split, runs multi-turn campaigns, and trains
the policy with TRL's GRPOTrainer. JSONL traces of every turn are written
under `training.log_dir`.

Usage:
    uvicorn drug_discovery_env.server.app:app --port 8000 &
    python -m drug_discovery_env.scripts.train [--config path/to/override.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from drug_discovery_env.config.runtime import load_settings
from drug_discovery_env.data_provider.dataset import load_dataset
from drug_discovery_env.training.grpo import train


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Live-rollout GRPO trainer")
    parser.add_argument("--config", type=Path, default=None, help="Optional override yaml")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Override settings.dataset.cache_path (useful on Kaggle/Colab)",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    if args.base_url:
        settings.training.base_url = args.base_url
    if args.model:
        settings.training.model_name = args.model
    if args.output_dir:
        settings.training.output_dir = args.output_dir
    if args.log_dir:
        settings.training.log_dir = args.log_dir
    if args.num_train_steps is not None:
        settings.training.num_train_steps = args.num_train_steps
    if args.group_size is not None:
        settings.training.group_size = args.group_size
    if args.cache_path:
        settings.dataset.cache_path = args.cache_path

    dataset = load_dataset(settings.dataset.cache_path, verbose=True)
    print(
        f"[train] base_url={settings.training.base_url}  model={settings.training.model_name}\n"
        f"        train_diseases={dataset.n_train}  test_diseases={dataset.n_test}\n"
        f"        steps={settings.training.num_train_steps}  group={settings.training.group_size}"
    )
    if dataset.n_train < 4:
        raise SystemExit(
            f"[train] train split has only {dataset.n_train} disease(s); "
            "rerun prepare_dataset with a larger --num-diseases (cache file: "
            f"{settings.dataset.cache_path})."
        )
    output_dir = train(settings, dataset, run_id=args.run_id)
    print(f"[train] checkpoint saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
