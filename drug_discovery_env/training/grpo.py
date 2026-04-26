"""Live-rollout GRPO trainer.

Single training pipeline:

  1. Build a HuggingFace dataset where each row carries one disease drawn from
     the cached train split.
  2. Stand up a TRL :class:`GRPOTrainer` with a custom multi-turn `rollout_func`
     that — for each prompt produced by the trainer — runs one full env
     campaign over HTTP against the FastAPI server and returns the
     concatenated turn outputs + the env's terminal reward (and the full
     reward breakdown).
  3. Per-component reward functions (`reward_total`, `reward_terminal`,
     `reward_stage`, `reward_budget`, `reward_reasoning`) read those fields so
     each component is visible in TRL logs.

All hyperparameters come from `settings.training` (`config/defaults.yaml`).
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List

import torch

from drug_discovery_env.client import DrugDiscoveryClient
from drug_discovery_env.config.settings import Settings
from drug_discovery_env.data_provider.dataset import DiseaseDataset
from drug_discovery_env.training.episode_logger import EpisodeLogger
from drug_discovery_env.training.prompting import (
    SYSTEM_PROMPT,
    action_from_text,
    initial_user_message,
    render_observation,
)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prompts(dataset: DiseaseDataset, num_prompts: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if not dataset.train:
        raise RuntimeError("Train split is empty; run prepare_dataset first.")
    rows: List[Dict[str, Any]] = []
    for _ in range(num_prompts):
        row = rng.choice(dataset.train)
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": initial_user_message(row.disease)},
                ],
                "disease": row.disease,
            }
        )
    return rows


def _disease_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if "Begin a new campaign for " in content:
                tail = content.split("Begin a new campaign for ", 1)[1]
                return tail.split(".", 1)[0].strip()
    return ""


def _make_rollout_func(
    base_url: str,
    logger: EpisodeLogger,
    max_turns: int,
) -> Callable[..., List[Dict[str, Any]]]:
    try:
        from trl.extras.openenv_utils import generate_rollout_completions  # type: ignore
    except Exception:  # pragma: no cover
        from trl.trainer.grpo_trainer import generate_rollout_completions  # type: ignore

    def rollout(trainer, prompts, **kwargs):  # noqa: ARG001
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            disease = _disease_from_prompt(prompt) or "Type 2 Diabetes"
            episode_id = logger.start_episode(disease=disease)
            completions: List[str] = []
            with DrugDiscoveryClient(base_url=base_url).sync() as env:
                step_result = env.reset(disease=disease)
                observation = step_result.observation
                turn = 0
                breakdown: Dict[str, float] = {}
                done = False
                terminated_reason = ""
                while not step_result.done and turn < max_turns:
                    turn += 1
                    user_msg = render_observation(observation)
                    messages = list(prompt) + [{"role": "user", "content": user_msg}]
                    rollout_out = generate_rollout_completions(trainer, [messages])
                    text_field = rollout_out.get("text", "")
                    if isinstance(text_field, list):
                        text = text_field[0] if text_field else ""
                    else:
                        text = str(text_field or "")
                    completions.append(text)
                    action = action_from_text(text)
                    step_result = env.step(action)
                    observation = step_result.observation
                    done = bool(step_result.done)
                    if observation.reward_breakdown is not None:
                        breakdown = observation.reward_breakdown.model_dump()
                    if done:
                        terminated_reason = str(
                            (observation.info or {}).get("terminated_reason") or ""
                        )
                        break

                terminal = float(breakdown.get("terminal_compound", 0.0))
                total = float(breakdown.get("total", float(step_result.reward or 0.0)))
                budget_total = float(observation.budget_total or 0.0)
                budget_frac = (
                    float(observation.budget_remaining) / budget_total if budget_total > 0 else 0.0
                )
                stage_completed = observation.stage in {"finished", "lead_validation"}
                oversight_violations = int(
                    (observation.info or {}).get("oversight_violations", 0) or 0
                )

                logger.end_episode(
                    episode_id,
                    disease=disease,
                    steps=observation.step_index,
                    terminal_reward=terminal,
                    total_reward=total,
                    stage_completed=stage_completed,
                    budget_remaining_frac=budget_frac,
                    oversight_violations=oversight_violations,
                    terminated_reason=terminated_reason or ("done" if done else "max_turns"),
                )

                results.append(
                    {
                        "completion": "\n---\n".join(completions),
                        "reward": total,
                        "terminal_compound": terminal,
                        "stage_progression": float(breakdown.get("stage_progression", 0.0)),
                        "budget_efficiency": float(breakdown.get("budget_efficiency", 0.0)),
                        "reasoning_depth": float(breakdown.get("reasoning_depth", 0.0)),
                        "process": float(breakdown.get("process", 0.0)),
                        "strategy": float(breakdown.get("strategy", 0.0)),
                        "oversight_penalty": float(breakdown.get("oversight_penalty", 0.0)),
                    }
                )
        return results

    return rollout


def reward_total(completions, **kwargs):  # noqa: ARG001
    return [float(c.get("reward", 0.0)) for c in completions]


def reward_terminal(completions, **kwargs):  # noqa: ARG001
    return [float(c.get("terminal_compound", 0.0)) for c in completions]


def reward_stage(completions, **kwargs):  # noqa: ARG001
    return [float(c.get("stage_progression", 0.0)) for c in completions]


def reward_budget(completions, **kwargs):  # noqa: ARG001
    return [float(c.get("budget_efficiency", 0.0)) for c in completions]


def reward_reasoning(completions, **kwargs):  # noqa: ARG001
    return [float(c.get("reasoning_depth", 0.0)) for c in completions]


def _load_model_and_tokenizer(settings: Settings):
    cfg = settings.training
    model = None
    tokenizer = None
    use_unsloth = cfg.use_unsloth and torch.cuda.is_available()
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=cfg.max_prompt_length + cfg.max_completion_length,
                load_in_4bit=cfg.load_in_4bit,
            )
            FastLanguageModel.for_training(model)
        except Exception as exc:
            print(f"[warn] Unsloth unavailable ({exc}); falling back to plain HF.")
            use_unsloth = False
    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        device = _select_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda" and getattr(model, "device", None) is None:
            model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def train(settings: Settings, dataset: DiseaseDataset, *, run_id: str | None = None) -> str:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    cfg = settings.training
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = EpisodeLogger(cfg.log_dir, run_id=run_id)
    rows = _build_prompts(
        dataset,
        num_prompts=max(cfg.num_train_steps * cfg.episodes_per_step * cfg.group_size, 16),
        seed=settings.dataset.seed,
    )
    hf_dataset = Dataset.from_list(rows)

    model, tokenizer = _load_model_and_tokenizer(settings)
    rollout_func = _make_rollout_func(
        base_url=cfg.base_url,
        logger=logger,
        max_turns=cfg.max_turns_per_episode,
    )

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=float(cfg.learning_rate),
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.group_size,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=float(cfg.warmup_ratio),
        beta=float(cfg.beta),
        epsilon=float(cfg.epsilon),
        num_train_epochs=1,
        max_steps=int(cfg.num_train_steps),
        logging_steps=1,
        save_steps=max(50, int(cfg.num_train_steps) // 10 or 50),
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_total,
            reward_terminal,
            reward_stage,
            reward_budget,
            reward_reasoning,
        ],
        rollout_func=rollout_func,
        train_dataset=hf_dataset,
        args=grpo_cfg,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir
