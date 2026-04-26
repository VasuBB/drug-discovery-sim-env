"""Single-turn GRPO trainer.

Each prompt asks the LLM to plan a full drug discovery campaign for one
disease. The completion is parsed for an ordered list of action JSON objects;
those actions are replayed against the FastAPI env server to obtain the
reward breakdown. Standard TRL :class:`GRPOTrainer` is used (no custom
``rollout_func``), so the trainer is compatible with every TRL >= 0.18.

All hyperparameters come from ``settings.training`` (``config/defaults.yaml``).
"""

from __future__ import annotations

import os
import random
import threading
from typing import Any, Dict, List

import torch

from drug_discovery_env.client import DrugDiscoveryClient
from drug_discovery_env.config.settings import Settings
from drug_discovery_env.data_provider.dataset import DiseaseDataset
from drug_discovery_env.training.episode_logger import EpisodeLogger, TurnRecord
from drug_discovery_env.training.prompting import (
    SYSTEM_PROMPT,
    action_from_payload,
    parse_action_sequence,
    plan_user_message,
)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prompts(
    dataset: DiseaseDataset, num_prompts: int, seed: int, max_actions: int
) -> List[Dict[str, Any]]:
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
                    {
                        "role": "user",
                        "content": plan_user_message(row.disease, max_actions),
                    },
                ],
                "disease": row.disease,
            }
        )
    return rows


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


class _EpisodePlayer:
    """Replay an LLM-produced action sequence against the env once and cache the result.

    ``play(disease, completion_text)`` is idempotent per (disease, completion);
    we memoize so the five per-component reward functions don't each spin up
    a fresh HTTP campaign.
    """

    def __init__(self, base_url: str, logger: EpisodeLogger, max_turns: int) -> None:
        self.base_url = base_url
        self.logger = logger
        self.max_turns = max_turns
        self._cache: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(disease: str, completion: str) -> str:
        return f"{disease}\u0001{hash(completion)}"

    def play(self, disease: str, completion: str) -> Dict[str, float]:
        key = self._key(disease, completion)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        actions = parse_action_sequence(completion, self.max_turns)
        episode_id = self.logger.start_episode(disease=disease)
        breakdown: Dict[str, float] = {}
        observation = None
        terminated_reason = ""
        done = False

        with DrugDiscoveryClient(base_url=self.base_url).sync() as env:
            try:
                step_result = env.reset(disease=disease)
            except Exception as exc:
                result = self._zero_breakdown()
                with self._lock:
                    self._cache[key] = result
                self.logger.end_episode(
                    episode_id,
                    disease=disease,
                    steps=0,
                    terminal_reward=0.0,
                    total_reward=0.0,
                    stage_completed=False,
                    budget_remaining_frac=0.0,
                    oversight_violations=0,
                    terminated_reason=f"reset_failed: {exc}",
                )
                return result

            observation = step_result.observation
            for turn_idx, payload in enumerate(actions, start=1):
                if step_result.done or turn_idx > self.max_turns:
                    break
                try:
                    action = action_from_payload(payload)
                    step_result = env.step(action)
                except Exception:
                    continue
                observation = step_result.observation
                if observation.reward_breakdown is not None:
                    breakdown = observation.reward_breakdown.model_dump()

                self.logger.log_turn(
                    episode_id,
                    TurnRecord(
                        step=int(getattr(observation, "step_index", turn_idx) or turn_idx),
                        stage=str(getattr(observation, "stage", "") or ""),
                        tool=str(payload.get("tool", "")),
                        params=dict(payload.get("params") or {}),
                        target_compound_id=payload.get("target_compound_id"),
                        reasoning=str(payload.get("reasoning", "")),
                        evidence_ids=[str(x) for x in (payload.get("evidence_ids") or [])],
                        model_raw_output=completion if turn_idx == 1 else "",
                        rendered_observation="",
                        tool_result=dict(getattr(observation, "last_result", {}) or {}),
                        subagent_messages=list(
                            getattr(observation, "subagent_messages", []) or []
                        ),
                        reward_breakdown=dict(breakdown),
                        budget_remaining=float(
                            getattr(observation, "budget_remaining", 0.0) or 0.0
                        ),
                        budget_total=float(
                            getattr(observation, "budget_total", 0.0) or 0.0
                        ),
                        last_tool_cost=float(
                            getattr(observation, "last_tool_cost", 0.0) or 0.0
                        ),
                        done=bool(step_result.done),
                        reward=float(step_result.reward or 0.0),
                    ),
                )

                done = bool(step_result.done)
                if done:
                    terminated_reason = str(
                        (observation.info or {}).get("terminated_reason") or ""
                    )
                    break

        terminal = float(breakdown.get("terminal_compound", 0.0))
        total = float(breakdown.get("total", 0.0))
        budget_total = float(getattr(observation, "budget_total", 0.0) or 0.0)
        budget_remaining = float(getattr(observation, "budget_remaining", 0.0) or 0.0)
        budget_frac = budget_remaining / budget_total if budget_total > 0 else 0.0
        stage = getattr(observation, "stage", "")
        stage_completed = stage in {"finished", "lead_validation"}
        oversight_violations = int(
            (getattr(observation, "info", {}) or {}).get("oversight_violations", 0) or 0
        )
        steps = int(getattr(observation, "step_index", 0) or 0)

        self.logger.end_episode(
            episode_id,
            disease=disease,
            steps=steps,
            terminal_reward=terminal,
            total_reward=total,
            stage_completed=stage_completed,
            budget_remaining_frac=budget_frac,
            oversight_violations=oversight_violations,
            terminated_reason=terminated_reason
            or ("done" if done else f"plan_exhausted ({len(actions)} actions)"),
        )

        result = {
            "total": total,
            "terminal_compound": terminal,
            "stage_progression": float(breakdown.get("stage_progression", 0.0)),
            "budget_efficiency": float(breakdown.get("budget_efficiency", 0.0)),
            "reasoning_depth": float(breakdown.get("reasoning_depth", 0.0)),
            "process": float(breakdown.get("process", 0.0)),
            "strategy": float(breakdown.get("strategy", 0.0)),
            "oversight_penalty": float(breakdown.get("oversight_penalty", 0.0)),
        }
        with self._lock:
            self._cache[key] = result
        return result

    @staticmethod
    def _zero_breakdown() -> Dict[str, float]:
        return {
            "total": 0.0,
            "terminal_compound": 0.0,
            "stage_progression": 0.0,
            "budget_efficiency": 0.0,
            "reasoning_depth": 0.0,
            "process": 0.0,
            "strategy": 0.0,
            "oversight_penalty": 0.0,
        }


def _resolve_diseases(disease, prompts, n: int) -> List[str]:
    if isinstance(disease, list) and disease:
        return [str(d) for d in disease]
    if isinstance(disease, str):
        return [disease] * n
    if isinstance(prompts, list):
        return [_disease_from_prompt(p) for p in prompts]
    return [""] * n


def _disease_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if "Plan a complete drug discovery campaign for " in content:
                tail = content.split(
                    "Plan a complete drug discovery campaign for ", 1
                )[1]
                return tail.split(".", 1)[0].strip()
    return ""


def _make_reward_funcs(player: _EpisodePlayer):
    def _scored(completions, prompts=None, disease=None, **_kwargs) -> List[Dict[str, float]]:
        diseases = _resolve_diseases(disease, prompts, len(completions))
        out: List[Dict[str, float]] = []
        for d, comp in zip(diseases, completions):
            out.append(player.play(d or "Type 2 Diabetes", _completion_text(comp)))
        return out

    def reward_total(completions, prompts=None, disease=None, **kwargs):
        return [r["total"] for r in _scored(completions, prompts, disease, **kwargs)]

    def reward_terminal(completions, prompts=None, disease=None, **kwargs):
        return [r["terminal_compound"] for r in _scored(completions, prompts, disease, **kwargs)]

    def reward_stage(completions, prompts=None, disease=None, **kwargs):
        return [r["stage_progression"] for r in _scored(completions, prompts, disease, **kwargs)]

    def reward_budget(completions, prompts=None, disease=None, **kwargs):
        return [r["budget_efficiency"] for r in _scored(completions, prompts, disease, **kwargs)]

    def reward_reasoning(completions, prompts=None, disease=None, **kwargs):
        return [r["reasoning_depth"] for r in _scored(completions, prompts, disease, **kwargs)]

    return [reward_total, reward_terminal, reward_stage, reward_budget, reward_reasoning]


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
            # 4-bit (and 8-bit) base weights are frozen; HuggingFace Trainer refuses
            # to fine-tune them directly. Attach LoRA adapters so GRPO updates the
            # adapters while keeping the quantized base in place.
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_gradient_checkpointing="unsloth",
                random_state=int(settings.dataset.seed),
            )
        except Exception as exc:
            print(f"[warn] Unsloth unavailable ({exc}); falling back to plain HF.")
            use_unsloth = False
    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        device = _select_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        if cfg.load_in_4bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as exc:
                print(f"[warn] 4-bit quantization unavailable ({exc}); using full precision.")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
        if device != "cuda" and getattr(model, "device", None) is None:
            model = model.to(device)
        if "quantization_config" in load_kwargs:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=16,
                        lora_alpha=16,
                        lora_dropout=0.0,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    ),
                )
            except Exception as exc:
                print(f"[warn] PEFT/LoRA wrap failed ({exc}); training full model.")
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
        max_actions=cfg.max_turns_per_episode,
    )
    hf_dataset = Dataset.from_list(rows)

    model, tokenizer = _load_model_and_tokenizer(settings)
    player = _EpisodePlayer(
        base_url=cfg.base_url,
        logger=logger,
        max_turns=cfg.max_turns_per_episode,
    )

    bf16_supported = bool(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    fp16_supported = torch.cuda.is_available() and not bf16_supported

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
        bf16=bf16_supported,
        fp16=fp16_supported,
        temperature=float(cfg.generation_temperature),
        top_p=float(cfg.generation_top_p),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=_make_reward_funcs(player),
        train_dataset=hf_dataset,
        args=grpo_cfg,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir
