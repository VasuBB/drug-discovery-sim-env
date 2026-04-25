"""GRPO training script for the Drug Discovery Sim Environment.

Trains Qwen2.5-3B-Instruct (Unsloth-loaded) with HF TRL's GRPOTrainer against
live rollouts in the drug discovery campaign environment. Designed to fit on
a free Colab T4 in <4h, but also runs on A100 with vLLM colocate enabled.

Usage:
    # 1) start the env
    uvicorn server.app:app --host 0.0.0.0 --port 8000 &

    # 2) random-policy baseline
    python training/train_model.py --baseline --baseline-episodes 10

    # 3) GRPO training
    python training/train_model.py --base-url http://localhost:8000
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

# Make repo importable when run as `python training/train_model.py`.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models import DrugDiscoveryAction  # noqa: E402

# Heavy ML imports are deferred so `--help` and `--baseline` work without GPU.

# ---------------------------------------------------------------------------
# Device setup — prefer CUDA, fall back to CPU.
# ---------------------------------------------------------------------------
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[device] CUDA not available — running on CPU.")


SYSTEM_PROMPT = """You are the Project Lead in a simulated drug discovery campaign.

Your goal: take a disease target through 5 stages (target_selection -> hit_id ->
hit_to_lead -> admet -> lead_validation) and nominate the best lead compound
before your budget or step cap runs out. You have a team of sub-agents
(chemist, toxicologist, budget, oversight) advising you each step. Listen to
toxicology BLOCK warnings. Conserve budget; docking is the most expensive
tool — reserve it for verified candidates.

You MUST reply with a single JSON object on one line, with these keys:
  "tool":  one of search_chembl, predict_binding_affinity, compute_admet,
           modify_molecule, run_docking, literature_search,
           delegate_to_subagent, advance_stage, abandon_compound,
           pause_and_review_all, request_subagent_summary
  "params": object of tool-specific arguments (e.g. {"target": "DPP4"} or
           {"smiles": "...", "instruction": "add fluorine"})
  "target_compound_id": the C### id of the active compound this acts on, or null
  "reasoning": one or two sentences justifying the decision in scientific terms
               (mention binding, ADMET, PAINS, Lipinski, scaffold, selectivity,
               novelty, budget, hypothesis as appropriate)

Do not output anything outside the JSON object."""


def render_observation(obs_dict: Dict[str, Any]) -> str:
    """Compact textual rendering of the observation for the LLM prompt."""
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


def parse_action(text: str) -> DrugDiscoveryAction:
    """Best-effort parse of the model's JSON output into a DrugDiscoveryAction."""
    m = _JSON_RE.search(text or "")
    if not m:
        return DrugDiscoveryAction(tool="pause_and_review_all", params={}, reasoning=text or "")
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return DrugDiscoveryAction(tool="pause_and_review_all", params={}, reasoning=text or "")
    return DrugDiscoveryAction(
        tool=str(obj.get("tool", "pause_and_review_all")),
        params=obj.get("params", {}) or {},
        target_compound_id=obj.get("target_compound_id"),
        reasoning=str(obj.get("reasoning", "")),
    )


# ---------------------------------------------------------------------------
# Untrained-baseline rollout (no model, just a random valid policy).
# ---------------------------------------------------------------------------


@dataclass
class RolloutSummary:
    total_reward: float
    breakdown: Dict[str, float]
    steps: int
    budget_remaining: float
    advanced_lead: Optional[str]
    terminated_reason: Optional[str]


def random_policy(obs_dict: Dict[str, Any]) -> DrugDiscoveryAction:
    """Cheap baseline: pick a tool roughly appropriate for the current stage."""
    import random as _r

    stage = obs_dict.get("stage")
    actives = obs_dict.get("active_compounds") or []
    target = obs_dict.get("selected_target")

    if stage == "target_selection":
        return DrugDiscoveryAction(
            tool="advance_stage",
            params={"target": _r.choice(["DPP4", "GLP1R", "ACE", "SERT", "EGFR", "BACE1"])},
            reasoning="Pick a plausible target.",
        )
    if not target:
        return DrugDiscoveryAction(tool="advance_stage", params={"target": "DPP4"}, reasoning="Need a target.")
    if not actives:
        return DrugDiscoveryAction(tool="search_chembl", params={"query": target, "max_results": 5})
    cid = _r.choice(actives)["id"]
    if stage == "hit_id":
        return _r.choice([
            DrugDiscoveryAction(tool="predict_binding_affinity", params={"target": target}, target_compound_id=cid),
            DrugDiscoveryAction(tool="advance_stage"),
        ])
    if stage == "hit_to_lead":
        return _r.choice([
            DrugDiscoveryAction(tool="modify_molecule", params={"instruction": "add methyl"}, target_compound_id=cid),
            DrugDiscoveryAction(tool="advance_stage"),
        ])
    if stage == "admet":
        return _r.choice([
            DrugDiscoveryAction(tool="compute_admet", target_compound_id=cid),
            DrugDiscoveryAction(tool="advance_stage"),
        ])
    if stage == "lead_validation":
        return _r.choice([
            DrugDiscoveryAction(tool="run_docking", params={"target": target}, target_compound_id=cid),
            DrugDiscoveryAction(tool="advance_stage", target_compound_id=cid),
        ])
    return DrugDiscoveryAction(tool="pause_and_review_all")


def play_one(env, policy, max_steps: int = 50) -> RolloutSummary:
    result = env.reset()
    obs = result.observation
    total_reward = 0.0
    breakdown: Dict[str, float] = {}
    steps = 0
    while not result.done and steps < max_steps:
        steps += 1
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        action = policy(obs_dict)
        result = env.step(action)
        obs = result.observation
        if result.reward is not None:
            total_reward = result.reward
        if obs.last_result and "reward_breakdown" in obs.last_result:
            breakdown = obs.last_result["reward_breakdown"]
    state = env.state()
    return RolloutSummary(
        total_reward=total_reward,
        breakdown=breakdown,
        steps=steps,
        budget_remaining=getattr(state, "budget_remaining", 0.0),
        advanced_lead=getattr(state, "advanced_compound_id", None),
        terminated_reason=getattr(state, "terminated_reason", None),
    )


def run_baseline(base_url: str, num_episodes: int, out_dir: Path) -> None:
    from client import DrugDiscoveryEnv

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[RolloutSummary] = []
    with DrugDiscoveryEnv(base_url=base_url).sync() as env:
        for ep in range(num_episodes):
            s = play_one(env, random_policy)
            summaries.append(s)
            print(
                f"[baseline {ep+1}/{num_episodes}] reward={s.total_reward:.3f}  "
                f"steps={s.steps}  budget_left={s.budget_remaining:.0f}  end={s.terminated_reason}"
            )

    avg = sum(s.total_reward for s in summaries) / max(1, len(summaries))
    print(f"\nBaseline mean reward over {len(summaries)} episodes: {avg:.3f}")
    (out_dir / "baseline_summaries.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# GRPO training (lazy imports — only loaded when invoked)
# ---------------------------------------------------------------------------


def build_dataset(num_prompts: int = 256):
    from datasets import Dataset  # type: ignore

    from server.scenario_generator import list_scenarios

    scenarios = list_scenarios()
    rows = []
    for i in range(num_prompts):
        sc = scenarios[i % len(scenarios)]
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Begin a new campaign for {sc.disease}. Stage: target_selection."},
            ],
            "disease": sc.disease,
        })
    return Dataset.from_list(rows)


def make_rollout_func(base_url: str, max_turns: int = 50):
    """Factory for the TRL rollout function bound to a live env URL."""
    from client import DrugDiscoveryEnv

    def rollout(trainer, prompts, **kwargs):
        # The exact import path for the TRL rollout helper has shifted between
        # TRL versions; try the common ones.
        try:
            from trl.extras.openenv_utils import generate_rollout_completions  # type: ignore
        except Exception:
            from trl.trainer.grpo_trainer import generate_rollout_completions  # type: ignore

        results = []
        for prompt in prompts:
            with DrugDiscoveryEnv(base_url=base_url).sync() as env:
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
                    action = parse_action(text)
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
                    "oversight_penalty": float(breakdown.get("oversight_penalty", 0.0)),
                })
        return results

    return rollout


def reward_total(completions, **kwargs) -> List[float]:
    return [float(c.get("reward", 0.0)) for c in completions]


def reward_terminal(completions, **kwargs) -> List[float]:
    return [float(c.get("terminal_compound", 0.0)) for c in completions]


def reward_stage(completions, **kwargs) -> List[float]:
    return [float(c.get("stage_progression", 0.0)) for c in completions]


def reward_budget(completions, **kwargs) -> List[float]:
    return [float(c.get("budget_efficiency", 0.0)) for c in completions]


def reward_reasoning(completions, **kwargs) -> List[float]:
    return [float(c.get("reasoning_depth", 0.0)) for c in completions]


def train_grpo(
    base_url: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "outputs/grpo",
    num_train_epochs: int = 1,
    learning_rate: float = 5e-6,
    use_unsloth: bool = True,
) -> None:
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
        except Exception as exc:
            print(f"[warn] Unsloth unavailable ({exc}); falling back to plain HF model.")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

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

    config = GRPOConfig(
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
        report_to=["trackio"],
    )

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
    parser = argparse.ArgumentParser(description="GRPO trainer for the drug discovery sim env.")
    parser.add_argument("--base-url", default=os.environ.get("ENV_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--no-unsloth", action="store_true", help="Skip Unsloth and use plain HF.")
    parser.add_argument("--baseline", action="store_true", help="Run random-policy baseline only.")
    parser.add_argument("--baseline-episodes", type=int, default=10)
    parser.add_argument("--baseline-out", default="outputs/baseline")
    args = parser.parse_args()

    if args.baseline:
        run_baseline(args.base_url, args.baseline_episodes, Path(args.baseline_out))
        return

    train_grpo(
        base_url=args.base_url,
        model_name=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_unsloth=not args.no_unsloth,
    )


if __name__ == "__main__":
    main()
