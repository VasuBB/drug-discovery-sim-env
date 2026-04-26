"""GRPO training against a live env via HTTP."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import torch

from drug_discovery_env.training.model_policy import SYSTEM_PROMPT, action_from_text, render_observation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[device] CUDA not available - running on CPU.")


def build_dataset(disease: str, num_prompts: int = 256):
    from datasets import Dataset

    rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Begin a new campaign for {disease}. Stage: target_selection.",
                },
            ],
            "disease": disease,
        }
        for _ in range(num_prompts)
    ]
    return Dataset.from_list(rows)


def make_rollout_func(base_url: str, disease: str, max_turns: int = 50):
    """Live rollout against the running env over HTTP."""
    from drug_discovery_env.client import DrugDiscoveryClient

    def rollout(trainer, prompts, **kwargs):  # noqa: ARG001
        try:
            from trl.extras.openenv_utils import generate_rollout_completions  # type: ignore
        except Exception:
            from trl.trainer.grpo_trainer import generate_rollout_completions  # type: ignore

        results = []
        for prompt in prompts:
            with DrugDiscoveryClient(base_url=base_url).sync() as env:
                step_result = env.reset(disease=disease)
                observation = step_result.observation
                turn = 0
                completions: List[str] = []
                while not step_result.done and turn < max_turns:
                    turn += 1
                    user_msg = render_observation(
                        observation.model_dump() if hasattr(observation, "model_dump") else observation.__dict__
                    )
                    messages = list(prompt) + [{"role": "user", "content": user_msg}]
                    rollout_out = generate_rollout_completions(trainer, [messages])
                    text = (
                        rollout_out["text"][0]
                        if isinstance(rollout_out.get("text"), list)
                        else rollout_out.get("text", "")
                    )
                    completions.append(text)
                    step_result = env.step(action_from_text(text))
                    observation = step_result.observation

                final_reward = float(step_result.reward or 0.0)
                breakdown = (observation.last_result or {}).get("reward_breakdown", {}) or {}
                results.append(
                    {
                        "completion": "\n---\n".join(completions),
                        "reward": final_reward,
                        "terminal_compound": float(breakdown.get("terminal_compound", 0.0)),
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


def reward_total(completions, **kwargs):
    return [float(c.get("reward", 0.0)) for c in completions]


def reward_terminal(completions, **kwargs):
    return [float(c.get("terminal_compound", 0.0)) for c in completions]


def reward_stage(completions, **kwargs):
    return [float(c.get("stage_progression", 0.0)) for c in completions]


def reward_budget(completions, **kwargs):
    return [float(c.get("budget_efficiency", 0.0)) for c in completions]


def reward_reasoning(completions, **kwargs):
    return [float(c.get("reasoning_depth", 0.0)) for c in completions]


def train_grpo(
    base_url: str,
    disease: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "outputs/grpo",
    num_train_epochs: int = 1,
    learning_rate: float = 5e-6,
    use_unsloth: bool = True,
    max_steps: Optional[int] = None,
) -> None:
    from trl import GRPOConfig, GRPOTrainer

    model = None
    tokenizer = None
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                load_in_4bit=True,
            )
            FastLanguageModel.for_training(model)
        except Exception as exc:
            print(f"[warn] Unsloth unavailable ({exc}); falling back to plain HF.")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE == "cuda" and getattr(model, "device", None) is None:
            model = model.to(DEVICE)

    dataset = build_dataset(disease=disease)
    rollout_func = make_rollout_func(base_url=base_url, disease=disease)

    cfg_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_generations": 2,
        "max_completion_length": 256,
        "max_prompt_length": 2048,
        "logging_steps": 1,
        "save_steps": 50,
        "report_to": [],
    }
    if max_steps is not None:
        cfg_kwargs["max_steps"] = int(max_steps)
    config = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_total, reward_terminal, reward_stage, reward_budget, reward_reasoning],
        rollout_func=rollout_func,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-rollout GRPO trainer")
    parser.add_argument("--base-url", default=os.environ.get("ENV_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--disease", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-unsloth", action="store_true", help="Skip Unsloth and use plain HF.")
    args = parser.parse_args()

    train_grpo(
        base_url=args.base_url,
        disease=args.disease,
        model_name=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_unsloth=not args.no_unsloth,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
