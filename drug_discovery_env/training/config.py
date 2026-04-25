from __future__ import annotations

from drug_discovery_env.config.settings import get_settings


def training_config() -> dict[str, object]:
    s = get_settings()
    return {
        "model": s.training.model_name,
        "learning_rate": s.training.learning_rate,
        "group_size": s.training.group_size,
        "lora_rank": s.training.lora_rank,
        "iterations": s.training.iterations,
        "max_completion_length": s.training.max_completion_length,
        "warmup_ratio": s.training.warmup_ratio,
        "kl_penalty_beta": s.training.kl_penalty_beta,
        "clip_epsilon": s.training.clip_epsilon,
    }
