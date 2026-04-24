from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from drug_discovery_env.training.config import training_config
from drug_discovery_env.training.rollout_generator import generate_rollouts


@dataclass
class TrainingRunReport:
    status: str
    config: dict[str, Any]
    num_samples: int
    reward_mean: float
    message: str


def run_training_dry_run(num_episodes: int = 3) -> TrainingRunReport:
    cfg = training_config()
    samples = generate_rollouts(num_episodes=num_episodes)
    rewards = [s.reward for s in samples]
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    return TrainingRunReport(
        status="dry_run",
        config=cfg,
        num_samples=len(samples),
        reward_mean=mean_reward,
        message="Generated rollout dataset and reward signals for GRPO.",
    )


def run_grpo_if_available() -> dict[str, Any]:
    cfg = training_config()
    try:
        import trl  # noqa: F401
    except Exception:
        report = run_training_dry_run()
        out = asdict(report)
        out["status"] = "dry_run_no_trl"
        out["message"] = "TRL not installed; completed dry-run data generation instead."
        return out

    # Hook point for real GRPOTrainer initialization in GPU runtime.
    report = run_training_dry_run()
    out = asdict(report)
    out["status"] = "trl_ready"
    out["message"] = "TRL detected; replace this hook with GRPOTrainer execution in GPU runtime."
    out["config"] = cfg
    return out
