"""GRPO trainer that rolls out against a live environment.

Combines:
  * the polished Unsloth + TRL GRPOTrainer setup from the `main` branch
    (`training/train_model.py`), and
  * the modular config + scenarios + composable reward of the `akshat-dev`
    branch.

Usage
-----
    # 1) start the OpenEnv server
    uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 &

    # 2) run a baseline (no GPU / model required)
    python -m drug_discovery_env.scripts.run_baseline --episodes 10

    # 3) train GRPO
    python -m drug_discovery_env.scripts.train_grpo_live \\
        --base-url http://localhost:8000 \\
        --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

from drug_discovery_env.core.scenarios import list_scenarios

SYSTEM_PROMPT = """You are the Project Lead in a simulated drug discovery campaign.

Your goal: take a disease through 5 stages (target_selection -> hit_id ->
hit_to_lead -> admet -> lead_validation) and nominate a high-quality lead
compound before your budget or step cap runs out. You have a team of
sub-agents (chemist, toxicologist, budget, oversight) advising you each
step. Listen to BLOCK warnings (especially toxicology). Conserve budget;
validate_compound is the most expensive tool -- reserve it for verified
candidates.

Reply with a single XML-like action on one line, with these tags:
  <reasoning>1-2 sentences of scientific justification (mention binding,
              ADMET, PAINS, Lipinski, scaffold, selectivity, novelty,
              budget, hypothesis as appropriate)</reasoning>
  <tool>one of select_target, search_compounds, predict_affinity,
        evaluate_admet, modify_molecule, synthesize, validate_compound,
        search_literature</tool>
  <params>{"smiles": "...", ...}</params>
  <evidence>pmid_1, pmid_2</evidence>   # optional, comma-separated
"""


def render_observation(obs: Any) -> str:
    """Compact textual observation for the LLM prompt."""

    info = getattr(obs, "info", {}) or {}
    meta = getattr(obs, "metadata", {}) or {}
    state_summary = getattr(obs, "state_summary", "")
    msgs = getattr(obs, "sub_agent_messages", {}) or {}
    flat_msgs: List[str] = []
    for tag, items in msgs.items():
        if tag == "messages":
            continue
        for line in items[:3]:
            flat_msgs.append(f"  [{tag}] {line}")
    text = [
        f"Stage: {info.get('stage_name', meta.get('stage_name', '?'))}  "
        f"Step: {info.get('step', 0)}",
        state_summary,
    ]
    tool_result = getattr(obs, "tool_result", None)
    if tool_result:
        text.append(f"Last tool result: {json.dumps(tool_result)[:240]}")
    if flat_msgs:
        text.append("Sub-agent messages:")
        text.extend(flat_msgs[:8])
    breakdown = getattr(obs, "reward_breakdown", None)
    if breakdown is not None:
        text.append(
            "Reward so far: "
            f"term={breakdown.terminal:.2f} proc={breakdown.process:.2f} "
            f"rea={breakdown.reasoning:.2f} strat={breakdown.strategy:.2f} "
            f"penalty={breakdown.oversight_penalty:.2f}"
        )
    return "\n".join(text)


_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


def looks_like_action(text: str) -> bool:
    return bool(_TOOL_RE.search(text or ""))


def build_dataset(num_prompts: int = 256):
    from datasets import Dataset  # type: ignore

    scenarios = list_scenarios()
    rows = []
    for i in range(num_prompts):
        sc = scenarios[i % len(scenarios)]
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Begin a new campaign for {sc.disease}. "
                        f"Stage: target_selection. Canonical target hint: "
                        f"{sc.canonical_target}."
                    ),
                },
            ],
            "disease": sc.disease,
        })
    return Dataset.from_list(rows)


def make_rollout_func(base_url: str, max_turns: int = 50):
    """Factory for the TRL rollout function bound to the live env URL."""

    from drug_discovery_env.client import DrugDiscoveryClient
    from drug_discovery_env.core.action_parser import ActionParser
    from drug_discovery_env.core.models import DrugDiscoveryAction

    parser = ActionParser({
        "select_target",
        "search_compounds",
        "predict_affinity",
        "evaluate_admet",
        "modify_molecule",
        "synthesize",
        "validate_compound",
        "search_literature",
    })

    def _safe_parse(text: str) -> DrugDiscoveryAction:
        try:
            return parser.parse(text)
        except Exception:
            # Fall back to a cheap noop-ish action so the rollout continues.
            return DrugDiscoveryAction(
                tool="search_literature",
                params={"query": "drug safety"},
                reasoning=text or "fallback",
            )

    def rollout(trainer, prompts, **kwargs):
        try:
            from trl.extras.openenv_utils import generate_rollout_completions  # type: ignore
        except Exception:
            from trl.trainer.grpo_trainer import generate_rollout_completions  # type: ignore

        results = []
        for prompt in prompts:
            with DrugDiscoveryClient(base_url=base_url).sync() as env:
                step_result = env.reset()
                obs = step_result.observation
                turn = 0
                full_text_parts: List[str] = []
                while not step_result.done and turn < max_turns:
                    turn += 1
                    user_msg = render_observation(obs)
                    messages = list(prompt) + [{"role": "user", "content": user_msg}]
                    rollout_out = generate_rollout_completions(trainer, [messages])
                    text = (
                        rollout_out["text"][0]
                        if isinstance(rollout_out.get("text"), list)
                        else rollout_out.get("text", "")
                    )
                    full_text_parts.append(text)
                    action = _safe_parse(text)
                    step_result = env.step(action)
                    obs = step_result.observation
                final_reward = step_result.reward if step_result.reward is not None else 0.0
                breakdown = getattr(obs, "reward_breakdown", None)
                bd = breakdown.to_dict() if breakdown else {}
                results.append({
                    "completion": "\n---\n".join(full_text_parts),
                    "reward": float(final_reward),
                    "terminal": float(bd.get("terminal", 0.0)),
                    "process": float(bd.get("process", 0.0)),
                    "reasoning": float(bd.get("reasoning", 0.0)),
                    "strategy": float(bd.get("strategy", 0.0)),
                    "oversight_penalty": float(bd.get("oversight_penalty", 0.0)),
                })
        return results

    return rollout


def reward_total(completions, **_):
    return [float(c.get("reward", 0.0)) for c in completions]


def reward_terminal(completions, **_):
    return [float(c.get("terminal", 0.0)) for c in completions]


def reward_process(completions, **_):
    return [float(c.get("process", 0.0)) for c in completions]


def reward_reasoning(completions, **_):
    return [float(c.get("reasoning", 0.0)) for c in completions]


def reward_strategy(completions, **_):
    return [float(c.get("strategy", 0.0)) for c in completions]


def train(
    base_url: str,
    model_name: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    use_unsloth: bool,
) -> None:
    import torch  # noqa: F401

    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    model = None
    tokenizer = None
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                load_in_4bit=True,
            )
            FastLanguageModel.for_training(model)
        except Exception as exc:  # pragma: no cover -- env-dependent
            print(f"[warn] Unsloth unavailable ({exc}); using plain HF model.")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = build_dataset()
    rollout_func = make_rollout_func(base_url=base_url)

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=2,
        max_completion_length=256,
        max_prompt_length=2048,
        logging_steps=1,
        save_steps=50,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_total,
            reward_terminal,
            reward_process,
            reward_reasoning,
            reward_strategy,
        ],
        rollout_func=rollout_func,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO live-rollout trainer.")
    parser.add_argument("--base-url", default=os.environ.get("ENV_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--no-unsloth", action="store_true")
    args = parser.parse_args()

    train(
        base_url=args.base_url,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_unsloth=not args.no_unsloth,
    )


if __name__ == "__main__":
    sys.exit(main())
