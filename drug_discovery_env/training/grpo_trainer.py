"""GRPO orchestration — dry-run + offline-dataset path used by run_training_experiment.

For *live-rollout* GRPO against a running env, see scripts/train_grpo_live.py.
This module is the offline path: generate heuristic rollouts, build a TRL
dataset where the reward column is the env's *actual* terminal reward (not a
keyword-counter), and either return a dry-run report or kick off TRL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from drug_discovery_env.config.settings import DataSourceMode
from drug_discovery_env.training.config import training_config
from drug_discovery_env.training.rollout_generator import generate_rollouts


@dataclass
class TrainingRunReport:
    status: str
    config: Dict[str, Any]
    num_samples: int
    reward_mean: float
    message: str


def run_training_dry_run(
    num_episodes: int = 3,
    *,
    data_mode: DataSourceMode = DataSourceMode.LIVE_ONLY,
    disease: str,
) -> TrainingRunReport:
    cfg = training_config()
    samples = generate_rollouts(num_episodes=num_episodes, data_mode=data_mode, disease=disease)
    rewards = [s.reward for s in samples]
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    return TrainingRunReport(
        status="dry_run",
        config=cfg,
        num_samples=len(samples),
        reward_mean=mean_reward,
        message="Generated rollout dataset and reward signals for GRPO.",
    )


def _build_training_dataset(
    num_episodes: int,
    *,
    data_mode: DataSourceMode = DataSourceMode.LIVE_ONLY,
    disease: str,
):
    from datasets import Dataset

    samples = generate_rollouts(num_episodes=num_episodes, data_mode=data_mode, disease=disease)
    rows = [
        {
            "prompt": s.prompt,
            "completion": s.completion,
            "reward": float(s.reward),
        }
        for s in samples
    ]
    return Dataset.from_list(rows), samples


def _make_env_grounded_reward_func(samples_by_prompt: Dict[str, float]):
    """Reward function bound to the env's reward, not a keyword counter.

    During offline GRPO we don't have the env in the loop, so we look up the
    env's reward by prompt-string match and hand it back unchanged. This keeps
    the reward signal aligned with what the agent will actually be evaluated on.
    """

    def reward_func(completions, prompts=None, **kwargs):  # noqa: ARG001
        if prompts is None:
            return [0.0 for _ in completions]
        return [float(samples_by_prompt.get(str(p), 0.0)) for p in prompts]

    return reward_func


def run_grpo_if_available(
    *,
    enable_actual_training: bool = False,
    model_name_override: str | None = None,
    num_episodes: int = 3,
    data_mode: DataSourceMode = DataSourceMode.LIVE_ONLY,
    disease: str,
    device: str = "auto",
    output_dir: str = "outputs/grpo",
    max_train_steps: int = 20,
) -> Dict[str, Any]:
    cfg = training_config()
    model_name = model_name_override or str(cfg["model"])

    try:
        import trl  # noqa: F401
    except Exception:
        report = run_training_dry_run(num_episodes=num_episodes, data_mode=data_mode, disease=disease)
        out = asdict(report)
        out["status"] = "dry_run_no_trl"
        out["message"] = "TRL not installed; completed dry-run data generation instead."
        return out

    dataset, samples = _build_training_dataset(num_episodes=num_episodes, data_mode=data_mode, disease=disease)
    rewards = [float(x.reward) for x in samples]
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0

    base = {
        "status": "trl_ready",
        "config": cfg,
        "model": model_name,
        "data_mode": data_mode.value,
        "disease": disease,
        "num_samples": len(samples),
        "reward_mean": mean_reward,
        "message": "TRL detected and dataset prepared.",
    }

    if not enable_actual_training:
        base["message"] = "TRL detected and GRPO dataset prepared. Set enable_actual_training=True to run trainer."
        return base

    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        base["status"] = "trl_missing_components"
        base["message"] = f"TRL available but GRPO components missing: {exc}"
        return base

    import torch

    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = requested

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ = AutoModelForCausalLM  # validate import

    samples_by_prompt = {s.prompt: s.reward for s in samples}
    reward_func = _make_env_grounded_reward_func(samples_by_prompt)

    import inspect

    num_generations = max(1, min(int(cfg["group_size"]), 2))
    raw_cfg: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(cfg["learning_rate"]),
        "max_completion_length": int(cfg["max_completion_length"]),
        "num_generations": num_generations,
        "generation_batch_size": num_generations,
        "per_device_train_batch_size": max(1, num_generations),
        "gradient_accumulation_steps": 1,
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "beta": float(cfg["kl_penalty_beta"]),
        "epsilon": float(cfg["clip_epsilon"]),
        "num_train_epochs": 1,
        "max_steps": max_train_steps,
        "logging_steps": 1,
        "save_steps": 100,
        "report_to": [],
        "use_cpu": resolved_device == "cpu",
    }
    sig = inspect.signature(GRPOConfig.__init__)
    supported = set(sig.parameters.keys())
    grpo_cfg = GRPOConfig(**{k: v for k, v in raw_cfg.items() if k in supported})

    trainer = GRPOTrainer(
        model=model_name,
        args=grpo_cfg,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    train_output = trainer.train()

    base["status"] = "trl_trained"
    base["device"] = resolved_device
    base["train_metrics"] = train_output.metrics if hasattr(train_output, "metrics") else {}
    base["log_history"] = getattr(getattr(trainer, "state", None), "log_history", [])
    base["output_dir"] = output_dir
    base["message"] = "GRPO trainer run completed."
    return base


__all__: List[str] = [
    "TrainingRunReport",
    "run_grpo_if_available",
    "run_training_dry_run",
]
