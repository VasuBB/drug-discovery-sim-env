"""Run a train-vs-baseline experiment and save reproducible plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from drug_discovery_env.config.settings import DataSourceMode
from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.grpo_trainer import run_grpo_if_available
from drug_discovery_env.training.model_policy import TransformersToolPolicy
from drug_discovery_env.training.policy_evaluator import evaluate_in_process


def model_mean_reward(
    model_name_or_path: str,
    *,
    disease: str,
    episodes: int,
    device: str,
) -> float:
    env = DrugDiscoveryEnv()
    policy = TransformersToolPolicy(model_name_or_path, device=device)
    metrics = evaluate_in_process(
        env=env,
        disease=disease,
        episodes=episodes,
        choose_action=policy.next_action,
    )
    return sum(m.total_reward for m in metrics) / len(metrics)


def plot_curves(log_history: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[int] = []
    losses: list[float] = []
    rewards: list[tuple[int, float]] = []
    for idx, row in enumerate(log_history):
        step = row.get("step", idx)
        if "loss" in row:
            steps.append(step)
            losses.append(float(row["loss"]))
        if "reward" in row:
            rewards.append((step, float(row["reward"])))

    loss_path = out_dir / "loss_curve.png"
    reward_path = out_dir / "reward_curve.png"

    if steps and losses:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, losses, label="training loss")
        plt.xlabel("training step")
        plt.ylabel("loss")
        plt.title("GRPO Training Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()

    if rewards:
        r_steps = [x[0] for x in rewards]
        r_vals = [x[1] for x in rewards]
        plt.figure(figsize=(7, 4))
        plt.plot(r_steps, r_vals, label="training reward")
        plt.xlabel("training step")
        plt.ylabel("reward")
        plt.title("GRPO Training Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_path, dpi=150)
        plt.close()

    return {"loss_curve": str(loss_path), "reward_curve": str(reward_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-vs-baseline experiment and save plots")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", type=str, default="artifacts/training")
    parser.add_argument("--max-train-steps", type=int, default=10)
    args = parser.parse_args()

    data_mode = DataSourceMode.LIVE_ONLY
    baseline = model_mean_reward(
        args.model,
        disease=args.disease,
        episodes=args.eval_episodes,
        device=args.device,
    )

    result = run_grpo_if_available(
        enable_actual_training=True,
        model_name_override=args.model,
        num_episodes=args.episodes,
        data_mode=data_mode,
        disease=args.disease,
        device=args.device,
        output_dir=args.out_dir,
        max_train_steps=args.max_train_steps,
    )

    log_history = result.get("log_history", [])
    paths = plot_curves(log_history, Path(args.out_dir))

    trained_reward = float(result.get("reward_mean", 0.0))
    trained_eval_status = "not_evaluated"
    if result.get("status") == "trl_trained":
        trained_reward = model_mean_reward(
            args.out_dir,
            disease=args.disease,
            episodes=args.eval_episodes,
            device=args.device,
        )
        trained_eval_status = "evaluated_checkpoint"

    compare_path = Path(args.out_dir) / "baseline_vs_trained.png"
    plt.figure(figsize=(6, 4))
    plt.bar(["base model", "trained model"], [baseline, trained_reward])
    plt.ylabel("mean reward")
    plt.title("Base Model vs Trained Model")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150)
    plt.close()

    summary = {
        "base_model_mean_reward": baseline,
        "trained_mean_reward": trained_reward,
        "trained_eval_status": trained_eval_status,
        "comparison_plot": str(compare_path),
        **paths,
        "run_result": result,
    }
    summary_path = Path(args.out_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
