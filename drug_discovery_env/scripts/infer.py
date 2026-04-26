"""Inference entrypoint — runs a trained checkpoint on a single (possibly
unseen) disease and prints / saves the campaign trace.

Usage:
    python -m drug_discovery_env.scripts.infer \
        --checkpoint outputs/grpo \
        --disease "Alzheimer disease"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from drug_discovery_env.config.runtime import load_settings, resolve_path
from drug_discovery_env.training.episode_logger import EpisodeLogger
from drug_discovery_env.training.policy import TransformersToolPolicy
from drug_discovery_env.training.rollout import run_episode


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "disease"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Single-disease inference")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs/infer")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    base_url = args.base_url or settings.training.base_url

    out_dir = resolve_path(args.out_dir)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = TransformersToolPolicy(
        args.checkpoint,
        device=args.device,
        max_new_tokens=settings.training.max_new_tokens_per_turn,
        temperature=settings.training.generation_temperature,
        top_p=settings.training.generation_top_p,
    )
    logger = EpisodeLogger(log_dir, run_id=args.run_id)

    print(f"[infer] disease={args.disease}  checkpoint={args.checkpoint}  base_url={base_url}")
    result = run_episode(
        base_url=base_url,
        disease=args.disease,
        generate_text=policy.generate_text,
        logger=logger,
        max_turns=settings.training.max_turns_per_episode,
    )

    summary = {
        "disease": args.disease,
        "episode_id": result.episode_id,
        "nominated_smiles": result.nominated_smiles,
        "nominated_compound_id": result.nominated_compound_id,
        "nominated_admet": result.nominated_admet,
        "final_stage": result.final_stage,
        "stage_completed": result.stage_completed,
        "terminal_reward": result.terminal_reward,
        "total_reward": result.total_reward,
        "reward_breakdown": result.reward_breakdown,
        "budget_remaining": result.budget_remaining,
        "budget_total": result.budget_total,
        "oversight_violations": result.oversight_violations,
        "steps": result.steps,
        "terminated_reason": result.terminated_reason,
        "reasoning_trace": [
            {
                "step": idx + 1,
                "tool": a.tool,
                "params": a.params,
                "target_compound_id": a.target_compound_id,
                "reasoning": a.reasoning,
            }
            for idx, a in enumerate(result.turn_actions)
        ],
    }

    summary_path = out_dir / f"{_slug(args.disease)}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[infer] summary written to {summary_path}")
    print(json.dumps({k: summary[k] for k in (
        "disease",
        "nominated_smiles",
        "nominated_compound_id",
        "final_stage",
        "stage_completed",
        "terminal_reward",
        "total_reward",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
