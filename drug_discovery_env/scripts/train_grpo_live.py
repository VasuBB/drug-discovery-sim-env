"""GRPO training against a *live* env via HTTP.

Pattern: launch the FastAPI server in a separate process, then run this script.
Rollouts call the LLM, parse its action, step the live env, and use the env's
*actual* reward as the GRPO signal — no keyword counter, no proxy.

  uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port 8000 &
  python -m drug_discovery_env.scripts.train_grpo_live --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Heavy ML imports are deferred so --help works without GPU.
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[device] CUDA not available — running on CPU.")


SYSTEM_PROMPT = """You are the Project Lead in a simulated drug discovery campaign.

Goal: take a disease through 5 stages (target_selection -> hit_id ->
hit_to_lead -> admet -> lead_validation) and nominate the best lead compound
before your budget or step cap runs out. You have a team of sub-agents
(chemist, toxicologist, budget, oversight) advising you each step. Listen to
toxicology BLOCK warnings. Conserve budget; validate_compound is the most
expensive tool — reserve it for verified candidates.

You MUST reply with a single JSON object on one line, with these keys:
  "tool":  one of select_target, search_compounds, predict_affinity,
           evaluate_admet, modify_molecule, synthesize, validate_compound,
           search_literature, advance_stage, abandon_compound,
           pause_and_review_all, delegate_to_subagent, request_subagent_summary
  "params": object of tool-specific arguments (e.g. {"target": "DPP4"} or
           {"smiles": "...", "instruction": "add fluorine"})
  "target_compound_id": the C### id of the active compound this acts on, or null
  "reasoning": one or two sentences justifying the decision in scientific terms
               (mention binding, ADMET, PAINS, Lipinski, scaffold, selectivity,
               novelty, budget, hypothesis, uncertainty, tradeoff)

Do not output anything outside the JSON object."""


def render_observation(obs_dict: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(
        f"Stage: {obs_dict.get('stage')}  Step: {obs_dict.get('step_index')}/{obs_dict.get('max_steps')}"
    )
    lines.append(
        f"Disease: {obs_dict.get('disease')}  Target: {obs_dict.get('selected_target')}"
    )
    lines.append(
        f"Budget: {obs_dict.get('budget_remaining', 0):.0f}/{obs_dict.get('budget_total', 0):.0f}"
        f"  (last cost {obs_dict.get('last_tool_cost', 0):.0f})"
    )
    if obs_dict.get("last_tool"):
        lines.append(
            f"Last tool: {obs_dict['last_tool']} -> "
            f"{json.dumps(obs_dict.get('last_result', {}))[:240]}"
        )
    msgs = obs_dict.get("subagent_messages") or []
    if msgs:
        lines.append("Sub-agent messages:")
        for m in msgs[:8]:
            lines.append(f"  [{m['agent']}/{m['severity']}] {m['message']}")
    actives = obs_dict.get("active_compounds") or []
    if actives:
        lines.append(f"Active compounds ({len(actives)}):")
        for c in actives[:8]:
            admet = c.get("admet") or {}
            smiles = (c.get("smiles") or "")[:40]
            lines.append(
                f"  {c.get('id')}  smiles={smiles}"
                f"  aff={c.get('binding_affinity_nM')}"
                f"  dock={c.get('docking_score')}"
                f"  ro5={admet.get('ro5_pass')} pains={admet.get('pains')} tox={admet.get('tox_score')}"
            )
    lines.append(f"Status: {obs_dict.get('message','')}")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action_text(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text or "")
    if not m:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"tool": "pause_and_review_all", "params": {}, "reasoning": text or ""}


def build_dataset(num_prompts: int = 256):
    from datasets import Dataset

    from drug_discovery_env.core.scenarios import list_scenarios

    scenarios = list_scenarios()
    rows = []
    for i in range(num_prompts):
        sc = scenarios[i % len(scenarios)]
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Begin a new campaign for {sc.disease}. Stage: target_selection.",
                },
            ],
            "disease": sc.disease,
        })
    return Dataset.from_list(rows)


def make_rollout_func(base_url: str, max_turns: int = 50):
    """Live rollout against the running env over HTTP."""
    from drug_discovery_env.client import DrugDiscoveryClient
    from drug_discovery_env.core.models import DrugDiscoveryAction

    def rollout(trainer, prompts, **kwargs):  # noqa: ARG001
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
                    user_msg = render_observation(
                        obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
                    )
                    messages = list(prompt) + [{"role": "user", "content": user_msg}]
                    rollout_out = generate_rollout_completions(trainer, [messages])
                    text = (
                        rollout_out["text"][0]
                        if isinstance(rollout_out.get("text"), list)
                        else rollout_out.get("text", "")
                    )
                    full_text_parts.append(text)
                    obj = parse_action_text(text)
                    action = DrugDiscoveryAction(
                        tool=str(obj.get("tool", "pause_and_review_all")),
                        params=obj.get("params", {}) or {},
                        target_compound_id=obj.get("target_compound_id"),
                        reasoning=str(obj.get("reasoning", "")),
                    )
                    step_result = env.step(action)
                    obs = step_result.observation
                final_reward = step_result.reward if step_result.reward is not None else 0.0
                breakdown = (obs.last_result or {}).get("reward_breakdown", {}) or {}
                results.append({
                    "completion": "\n---\n".join(full_text_parts),
                    "reward": float(final_reward),
                    "terminal_compound": float(breakdown.get("terminal_compound", 0.0)),
                    "stage_progression": float(breakdown.get("stage_progression", 0.0)),
                    "budget_efficiency": float(breakdown.get("budget_efficiency", 0.0)),
                    "reasoning_depth": float(breakdown.get("reasoning_depth", 0.0)),
                    "process": float(breakdown.get("process", 0.0)),
                    "strategy": float(breakdown.get("strategy", 0.0)),
                    "oversight_penalty": float(breakdown.get("oversight_penalty", 0.0)),
                })
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

    dataset = build_dataset()
    rollout_func = make_rollout_func(base_url=base_url)

    cfg_kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-unsloth", action="store_true", help="Skip Unsloth and use plain HF.")
    args = parser.parse_args()

    train_grpo(
        base_url=args.base_url,
        model_name=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_unsloth=not args.no_unsloth,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
