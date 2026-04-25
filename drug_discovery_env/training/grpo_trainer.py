from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from drug_discovery_env.config.settings import DataSourceMode
from drug_discovery_env.training.config import training_config
from drug_discovery_env.training.rollout_generator import generate_rollouts


@dataclass
class TrainingRunReport:
    status: str
    config: dict[str, Any]
    num_samples: int
    reward_mean: float
    message: str


def run_training_dry_run(
    num_episodes: int = 3,
    *,
    data_mode: DataSourceMode = DataSourceMode.HYBRID,
    disease: str = "Type 2 Diabetes",
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
    data_mode: DataSourceMode = DataSourceMode.HYBRID,
    disease: str = "Type 2 Diabetes",
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


def run_grpo_if_available(
    *,
    enable_actual_training: bool = False,
    model_name_override: str | None = None,
    num_episodes: int = 3,
    data_mode: DataSourceMode = DataSourceMode.HYBRID,
    disease: str = "Type 2 Diabetes",
    device: str = "auto",
    output_dir: str = "outputs/grpo",
    max_train_steps: int = 20,
) -> dict[str, Any]:
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

    if resolved_device == "cuda":
        torch_dtype = torch.float16
    elif resolved_device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _ = AutoModelForCausalLM  # kept imported to validate transformers install

    def reward_func(completions, **kwargs):
        rewards: list[float] = []
        for completion in completions:
            text = str(completion).lower()
            score = 0.0
            if "<tool>" in text:
                score += 0.25
            if "<params>" in text:
                score += 0.20
            if "uncertain" in text or "risk" in text:
                score += 0.20
            if "admet" in text or "safety" in text:
                score += 0.20
            if "selectiv" in text or "potency" in text:
                score += 0.15
            rewards.append(min(1.0, score))
        return rewards

    import inspect

    num_generations = max(1, min(int(cfg["group_size"]), 2))
    raw_cfg: dict[str, Any] = {
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

    # Keep runtime lightweight in local machines.
    train_output = trainer.train()

    base["status"] = "trl_trained"
    base["device"] = resolved_device
    base["train_metrics"] = train_output.metrics if hasattr(train_output, "metrics") else {}
    base["log_history"] = getattr(getattr(trainer, "state", None), "log_history", [])
    base["output_dir"] = output_dir
    base["message"] = "GRPO trainer run completed."
    return base
