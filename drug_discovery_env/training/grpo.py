"""Single-turn GRPO trainer.

Each prompt asks the LLM to plan a full drug discovery campaign for one
disease. The completion is parsed for an ordered list of action JSON objects;
those actions are replayed against the FastAPI env server to obtain the
reward breakdown. Standard TRL :class:`GRPOTrainer` is used (no custom
``rollout_func``), so the trainer is compatible with every TRL >= 0.18.

All hyperparameters come from ``settings.training`` (``config/defaults.yaml``).
"""

from __future__ import annotations

import csv
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        episode_id = self.logger.start_episode(
            disease=disease, model_completion=completion
        )
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

        nominated_smiles, nominated_compound_id, nominated_admet = self._nominated(observation)

        self.logger.end_episode(
            episode_id,
            disease=disease,
            steps=steps,
            terminal_reward=terminal,
            total_reward=total,
            stage_completed=stage_completed,
            budget_remaining_frac=budget_frac,
            oversight_violations=oversight_violations,
            nominated_compound_id=nominated_compound_id,
            nominated_smiles=nominated_smiles,
            nominated_admet=nominated_admet,
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

    @staticmethod
    def _nominated(observation: Any) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Pick the lead compound from the final observation (matches rollout.py)."""

        if observation is None:
            return None, None, {}
        compound_id = getattr(observation, "advanced_compound_id", None)
        smiles: Optional[str] = None
        admet: Dict[str, Any] = {}
        actives = getattr(observation, "active_compounds", None) or []
        for compound in actives:
            if compound.get("id") == compound_id:
                smiles = compound.get("smiles")
                admet = compound.get("admet") or {}
                break
        if smiles is None and actives:
            best = max(
                actives,
                key=lambda c: (
                    float(c.get("potency", 0.0))
                    + float(c.get("selectivity", 0.0))
                    + float(c.get("safety", 0.0))
                    + float(c.get("developability", 0.0))
                )
                / 4.0,
            )
            smiles = best.get("smiles")
            compound_id = best.get("id")
            admet = best.get("admet") or {}
        return smiles, compound_id, admet


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


class _PlotsLogger:
    """Render loss / reward / per-reward-func plots after every optimizer step.

    Wrapped by a :class:`~transformers.TrainerCallback` subclass built inside
    :func:`train` so importing this module doesn't require ``transformers``.
    Every callback fire we append the new metrics to ``metrics.csv`` and
    re-render PNGs into ``<plots_dir>/`` — this is the live evidence of
    training (loss + per-reward-component curves) without needing to wait for
    the run to finish.
    """

    def __init__(self, plots_dir: str | Path, metrics_csv: str | Path) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = Path(metrics_csv)
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, float]] = []
        self._columns: List[str] = ["step"]
        self._step_count = 0

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _ensure_columns(self, entry: Dict[str, float]) -> None:
        new = [k for k in entry.keys() if k not in self._columns]
        if new:
            self._columns.extend(new)
            with self.metrics_csv.open("w", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._columns)
                writer.writeheader()
                for row in self._history:
                    writer.writerow({k: row.get(k, "") for k in self._columns})

    def _append_csv(self, entry: Dict[str, float]) -> None:
        if not self.metrics_csv.exists() or self.metrics_csv.stat().st_size == 0:
            with self.metrics_csv.open("w", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._columns)
                writer.writeheader()
        self._ensure_columns(entry)
        with self.metrics_csv.open("a", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            writer.writerow({k: entry.get(k, "") for k in self._columns})

    def _render(self) -> None:
        if not self._history:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - matplotlib should be available
            print(f"[plot] matplotlib unavailable ({exc}); skipping render")
            return

        def _xy(key: str) -> tuple[List[float], List[float]]:
            xs: List[float] = []
            ys: List[float] = []
            for row in self._history:
                v = row.get(key)
                if v is None or not isinstance(v, (int, float)) or v != v:
                    continue
                xs.append(row["step"])
                ys.append(float(v))
            return xs, ys

        loss_x, loss_y = _xy("loss")
        if loss_y:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(loss_x, loss_y, linewidth=1.5, color="#c0392b")
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("loss")
            ax.set_title("GRPO training loss")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.plots_dir / "training_loss.png", dpi=150)
            plt.close(fig)

        reward_x, reward_y = _xy("reward")
        if reward_y:
            stds: List[float] = []
            for row in self._history:
                v = row.get("reward")
                if v is None or not isinstance(v, (int, float)) or v != v:
                    continue
                stds.append(float(row.get("reward_std", 0.0) or 0.0))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(reward_x, reward_y, linewidth=1.5, color="#2c7fb8", label="mean reward")
            if any(s > 0 for s in stds):
                lo = [m - s for m, s in zip(reward_y, stds)]
                hi = [m + s for m, s in zip(reward_y, stds)]
                ax.fill_between(
                    reward_x, lo, hi, color="#2c7fb8", alpha=0.15, label="+/-1 std"
                )
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("reward")
            ax.set_title("GRPO group reward per step")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.plots_dir / "training_reward.png", dpi=150)
            plt.close(fig)

        component_keys = sorted(
            {
                k
                for row in self._history
                for k in row.keys()
                if k.startswith("rewards/") and not k.endswith("/std")
            }
        )
        if component_keys:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            palette = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]
            for idx, key in enumerate(component_keys):
                xs, ys = _xy(key)
                if not xs:
                    continue
                label = key.replace("rewards/", "").replace("/mean", "")
                ax.plot(xs, ys, linewidth=1.4, color=palette[idx % len(palette)], label=label)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel("reward component")
            ax.set_title("Per-component reward functions")
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.plots_dir / "reward_components.png", dpi=150)
            plt.close(fig)

    def record(self, step: int, logs: Dict[str, Any]) -> None:
        if not logs:
            return
        entry: Dict[str, float] = {"step": int(step or 0)}
        for key, value in logs.items():
            if self._is_number(value):
                entry[key] = float(value)
        if len(entry) <= 1:
            return
        self._history.append(entry)
        self._step_count += 1
        try:
            self._append_csv(entry)
        except Exception as exc:  # pragma: no cover
            print(f"[plot] csv append failed: {exc}")
        try:
            self._render()
        except Exception as exc:  # pragma: no cover
            print(f"[plot] render failed: {exc}")


def _build_plots_callback(plots_dir: str | Path, metrics_csv: str | Path):
    """Build a HF TrainerCallback that drives :class:`_PlotsLogger` on every log."""

    from transformers import TrainerCallback

    plots = _PlotsLogger(plots_dir=plots_dir, metrics_csv=metrics_csv)

    class TrainingPlotsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
            plots.record(int(getattr(state, "global_step", 0) or 0), logs or {})

    return TrainingPlotsCallback(), plots


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

    plots_dir = Path(cfg.output_dir) / "plots"
    metrics_csv = Path(cfg.output_dir) / "metrics.csv"
    plots_callback, _plots_logger = _build_plots_callback(plots_dir, metrics_csv)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=_make_reward_funcs(player),
        train_dataset=hf_dataset,
        args=grpo_cfg,
        callbacks=[plots_callback],
    )
    print(f"[train] live plots will be written to {plots_dir}")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[train] final plots saved to {plots_dir}")
    return cfg.output_dir
