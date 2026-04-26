"""GRPO orchestration for local rollouts and offline readiness checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.config import training_config
from drug_discovery_env.training.model_policy import action_from_text
from drug_discovery_env.training.rollout_generator import generate_rollouts, normalize_diseases


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
    disease: str | None = None,
    diseases: Sequence[str] | None = None,
) -> TrainingRunReport:
    cfg = training_config()
    samples = generate_rollouts(
        num_episodes=num_episodes,
        data_mode=data_mode,
        disease=disease,
        diseases=diseases,
    )
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
    disease: str | None = None,
    diseases: Sequence[str] | None = None,
):
    from datasets import Dataset

    samples = generate_rollouts(
        num_episodes=num_episodes,
        data_mode=data_mode,
        disease=disease,
        diseases=diseases,
    )
    rows = [
        {
            "prompt": s.prompt,
            "completion": s.completion,
            "reward": float(s.reward),
            "disease": s.disease,
            "history": list(s.history),
        }
        for s in samples
    ]
    return Dataset.from_list(rows), samples


def _restore_env_before_step(
    history: Sequence[str],
    *,
    disease: str,
    settings: Settings,
    snapshot_cache: Dict[tuple[str, tuple[str, ...]], tuple[Any, str | None]],
) -> DrugDiscoveryEnv:
    key = (disease, tuple(history))
    if key not in snapshot_cache:
        env = DrugDiscoveryEnv(settings=settings)
        env.reset(disease=disease)
        for prior_action in history:
            env.step(action_from_text(prior_action))
        snapshot_cache[key] = (deepcopy(env._game_state), env._episode_id)

    state_snapshot, episode_id = snapshot_cache[key]
    restored = DrugDiscoveryEnv(settings=settings)
    restored._game_state = deepcopy(state_snapshot)
    restored._episode_id = episode_id
    return restored


def _make_replay_reward_func(settings: Settings):
    snapshot_cache: Dict[tuple[str, tuple[str, ...]], tuple[Any, str | None]] = {}

    def reward_func(
        completions,
        prompts=None,  # noqa: ARG001
        disease=None,
        history=None,
        log_metric=None,
        **kwargs,  # noqa: ARG001
    ):
        if disease is None or history is None:
            return [0.0 for _ in completions]

        rewards: list[float] = []
        for completion, sample_disease, sample_history in zip(completions, disease, history, strict=True):
            try:
                env = _restore_env_before_step(
                    list(sample_history),
                    disease=str(sample_disease),
                    settings=settings,
                    snapshot_cache=snapshot_cache,
                )
                pre_breakdown = env.reward_engine.compute(env._ensure_state(), terminal=False)
                observation = env.step(action_from_text(str(completion)))
                post_total = (
                    float(observation.reward_breakdown.total)
                    if observation.reward_breakdown is not None
                    else float(observation.reward or 0.0)
                )
                shaped = post_total - float(pre_breakdown.total)
                if observation.done and observation.reward is not None:
                    shaped += float(observation.reward)
            except Exception:
                shaped = 0.0
            rewards.append(shaped)

        if callable(log_metric) and rewards:
            log_metric("env_reward/replay_mean", sum(rewards) / len(rewards))
        return rewards

    return reward_func


def _make_debug_replay_reward_func(
    settings: Settings,
    *,
    debug_limit: int,
):
    snapshot_cache: Dict[tuple[str, tuple[str, ...]], tuple[Any, str | None]] = {}
    state = {"printed": 0}

    def reward_func(
        completions,
        prompts=None,
        disease=None,
        history=None,
        log_metric=None,
        **kwargs,  # noqa: ARG001
    ):
        if disease is None or history is None:
            return [0.0 for _ in completions]

        rewards: list[float] = []
        prompt_values = prompts or ["" for _ in completions]
        for prompt, completion, sample_disease, sample_history in zip(
            prompt_values, completions, disease, history, strict=True
        ):
            debug_payload: Dict[str, Any] = {
                "prompt": str(prompt),
                "completion": str(completion),
                "disease": str(sample_disease),
                "history_len": len(sample_history),
            }
            try:
                env = _restore_env_before_step(
                    list(sample_history),
                    disease=str(sample_disease),
                    settings=settings,
                    snapshot_cache=snapshot_cache,
                )
                pre_breakdown = env.reward_engine.compute(env._ensure_state(), terminal=False)
                parsed_action = action_from_text(str(completion))
                observation = env.step(parsed_action)
                post_total = (
                    float(observation.reward_breakdown.total)
                    if observation.reward_breakdown is not None
                    else float(observation.reward or 0.0)
                )
                shaped = post_total - float(pre_breakdown.total)
                if observation.done and observation.reward is not None:
                    shaped += float(observation.reward)
                debug_payload.update(
                    {
                        "parsed_action": parsed_action.model_dump(),
                        "tool_result": observation.last_result,
                        "message": observation.message,
                        "next_state": observation.state_summary,
                        "reward": shaped,
                    }
                )
            except Exception as exc:
                shaped = 0.0
                debug_payload["error"] = str(exc)
                debug_payload["reward"] = shaped
            rewards.append(shaped)

            if state["printed"] < debug_limit:
                state["printed"] += 1
                print(
                    "[training-debug] sample\n"
                    f"prompt:\n{debug_payload['prompt']}\n\n"
                    f"completion:\n{debug_payload['completion']}\n\n"
                    f"parsed_action: {debug_payload.get('parsed_action')}\n"
                    f"tool_result: {debug_payload.get('tool_result')}\n"
                    f"message: {debug_payload.get('message')}\n"
                    f"next_state: {debug_payload.get('next_state')}\n"
                    f"reward: {debug_payload['reward']}\n"
                    + (f"error: {debug_payload['error']}\n" if "error" in debug_payload else ""),
                    flush=True,
                )

        if callable(log_metric) and rewards:
            log_metric("env_reward/replay_mean", sum(rewards) / len(rewards))
        return rewards

    return reward_func


def run_grpo_if_available(
    *,
    enable_actual_training: bool = False,
    model_name_override: str | None = None,
    num_episodes: int = 3,
    data_mode: DataSourceMode = DataSourceMode.LIVE_ONLY,
    disease: str | None = None,
    diseases: Sequence[str] | None = None,
    device: str = "auto",
    output_dir: str = "outputs/grpo",
    max_train_steps: int = 20,
    debug_io: bool = False,
    debug_limit: int = 10,
) -> Dict[str, Any]:
    cfg = training_config()
    model_name = model_name_override or str(cfg["model"])
    disease_list = normalize_diseases(disease=disease, diseases=diseases)
    if not disease_list:
        raise ValueError("At least one disease is required")

    try:
        import trl  # noqa: F401
    except Exception:
        report = run_training_dry_run(
            num_episodes=num_episodes,
            data_mode=data_mode,
            diseases=disease_list,
        )
        out = asdict(report)
        out["status"] = "dry_run_no_trl"
        out["message"] = "TRL not installed; completed dry-run data generation instead."
        return out

    dataset, samples = _build_training_dataset(
        num_episodes=num_episodes,
        data_mode=data_mode,
        diseases=disease_list,
    )
    rewards = [float(x.reward) for x in samples]
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0

    base = {
        "status": "trl_ready",
        "config": cfg,
        "model": model_name,
        "data_mode": data_mode.value,
        "disease": disease_list[0],
        "diseases": disease_list,
        "num_samples": len(samples),
        "reward_mean": mean_reward,
        "message": "TRL detected and dataset prepared.",
    }

    if not enable_actual_training:
        base["message"] = "TRL detected and GRPO dataset prepared. Set enable_actual_training=True to run trainer."
        return base

    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer
    except Exception as exc:
        base["status"] = "trl_missing_components"
        base["message"] = f"TRL available but GRPO components missing: {exc}"
        return base

    import inspect
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
    tokenizer.padding_side = "left"

    env_settings = get_settings().model_copy(deep=True)
    env_settings.data.mode = DataSourceMode.LIVE_ONLY
    reward_func = (
        _make_debug_replay_reward_func(env_settings, debug_limit=debug_limit)
        if debug_io
        else _make_replay_reward_func(env_settings)
    )

    num_generations = max(2, min(int(cfg["group_size"]), 2))
    raw_cfg: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(cfg["learning_rate"]),
        "max_completion_length": min(int(cfg["max_completion_length"]), 256),
        "max_prompt_length": 2048,
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

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
    "_build_training_dataset",
    "_make_replay_reward_func",
]
