from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from drug_discovery_env.client import create_sync_client
from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction
from drug_discovery_env.training.grpo_trainer import run_grpo_if_available
from drug_discovery_env.utils.logging import get_logger, write_jsonl


def random_baseline_reward_local(num_episodes: int, data_mode: DataSourceMode) -> float:
    from drug_discovery_env.server.environment import DrugDiscoveryEnv

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = data_mode
    env = DrugDiscoveryEnv(settings=settings)
    tools = [
        "select_target",
        "search_compounds",
        "predict_affinity",
        "evaluate_admet",
        "search_literature",
        "validate_compound",
    ]
    rewards: list[float] = []
    for _ in range(max(1, num_episodes)):
        obs = env.reset()
        done = False
        while not done:
            gs = getattr(env, "_game_state", None)
            smiles = next(iter(gs.compound_ledger.keys()), None) if gs and gs.compound_ledger else None
            tool = random.choice(tools)
            params = {}
            if tool in {"predict_affinity", "evaluate_admet", "validate_compound"} and smiles:
                params = {"smiles": smiles, "assay_type": "biochemical"}
            if tool == "select_target":
                params = {"disease": "Type 2 Diabetes"}
            if tool == "search_literature":
                params = {"query": "drug discovery safety selectivity"}
            action = f"<reasoning>random baseline action</reasoning><tool>{tool}</tool><params>{json.dumps(params)}</params>"
            try:
                obs = env.step(action)
            except Exception:
                obs = env.step("<reasoning>fallback</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.5}</params>")
            rewards.append(float(obs.reward or 0.0))
            done = obs.done
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def random_baseline_reward_remote(num_episodes: int, server_url: str) -> float:
    tools = [
        "select_target",
        "search_compounds",
        "predict_affinity",
        "evaluate_admet",
        "search_literature",
        "validate_compound",
    ]
    rewards: list[float] = []
    for _ in range(max(1, num_episodes)):
        with create_sync_client(server_url) as client:
            obs = client.reset().observation
            done = False
            smiles: str | None = None
            while not done:
                tool = random.choice(tools)
                params: dict[str, str] = {}
                if tool in {"predict_affinity", "evaluate_admet", "validate_compound"} and smiles:
                    params = {"smiles": smiles, "assay_type": "biochemical"}
                if tool == "select_target":
                    params = {"disease": "Type 2 Diabetes"}
                if tool == "search_literature":
                    params = {"query": "drug discovery safety selectivity"}
                action = DrugDiscoveryAction(tool=tool, reasoning="random baseline action", params=params)
                try:
                    step = client.step(action)
                except Exception:
                    step = client.step(
                        DrugDiscoveryAction(
                            tool="search_compounds",
                            reasoning="fallback",
                            params={"min_qed": 0.5},
                        )
                    )
                obs = step.observation
                hits = (obs.tool_result or {}).get("hits", []) if isinstance(obs.tool_result, dict) else []
                if hits and isinstance(hits[0], dict) and hits[0].get("smiles"):
                    smiles = str(hits[0]["smiles"])
                rewards.append(float(step.reward if step.reward is not None else (obs.reward or 0.0)))
                done = bool(step.done)
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def plot_curves(log_history: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    losses = []
    rewards = []
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

    return {
        "loss_curve": str(loss_path),
        "reward_curve": str(reward_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-vs-baseline experiment and save plots")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--data-mode", choices=[m.value for m in DataSourceMode], default="local_only")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", type=str, default="artifacts/training")
    parser.add_argument("--max-train-steps", type=int, default=10)
    parser.add_argument("--server-url", type=str, default=None, help="Use running OpenEnv server via client")
    args = parser.parse_args()
    logger = get_logger("drug_discovery_env.training_experiment", log_file="artifacts/logs/training_experiment.log")

    data_mode = DataSourceMode(args.data_mode)

    baseline = (
        random_baseline_reward_remote(args.episodes, args.server_url)
        if args.server_url
        else random_baseline_reward_local(args.episodes, data_mode)
    )

    result = run_grpo_if_available(
        enable_actual_training=True,
        model_name_override=args.model,
        num_episodes=args.episodes,
        data_mode=data_mode,
        device=args.device,
        output_dir=args.out_dir,
        max_train_steps=args.max_train_steps,
        server_url=args.server_url,
    )

    log_history = result.get("log_history", [])
    paths = plot_curves(log_history, Path(args.out_dir))

    trained_reward = float(result.get("reward_mean", 0.0))
    compare_path = Path(args.out_dir) / "baseline_vs_trained.png"
    plt.figure(figsize=(6, 4))
    plt.bar(["baseline", "trained"], [baseline, trained_reward])
    plt.ylabel("mean reward")
    plt.title("Baseline vs Trained")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150)
    plt.close()

    summary = {
        "baseline_mean_reward": baseline,
        "trained_mean_reward": trained_reward,
        "comparison_plot": str(compare_path),
        **paths,
        "run_result": result,
    }

    summary_path = Path(args.out_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl("artifacts/logs/training_experiment.jsonl", summary)
    logger.info(
        "experiment complete baseline=%.4f trained=%.4f out_dir=%s",
        baseline,
        trained_reward,
        args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
