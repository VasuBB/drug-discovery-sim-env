"""Random-policy baseline runner.

The baseline plays valid stage-aware actions and gives the floor that GRPO
has to beat. Ported from the `main` branch's `training/train_model.py
--baseline`. Writes per-episode summaries to JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction
from drug_discovery_env.server.environment import DrugDiscoveryEnv


def random_policy(env: DrugDiscoveryEnv, last_obs) -> DrugDiscoveryAction:
    """Cheap stage-aware policy. No model required."""

    state = env._game_state  # noqa: SLF001 -- baseline introspection only
    stage = state.stage if state else 1
    compounds = list(state.compound_ledger.keys()) if state else []
    target_known = bool(state and state.target)

    if stage == 1 or not target_known:
        return DrugDiscoveryAction(
            tool="select_target",
            params={"disease": state.disease if state else "Type 2 Diabetes"},
            reasoning="Pick the strongest target for this disease before screening.",
        )
    if not compounds:
        return DrugDiscoveryAction(
            tool="search_compounds",
            params={"min_qed": 0.4},
            reasoning="Need a candidate pool to screen for binding affinity.",
        )
    smiles = random.choice(compounds)
    if stage == 2:
        return DrugDiscoveryAction(
            tool="predict_affinity",
            params={"smiles": smiles, "assay_type": "biochemical"},
            reasoning="Measure potency before optimisation; trade-off cost vs information.",
        )
    if stage == 3:
        return DrugDiscoveryAction(
            tool="modify_molecule",
            params={"smiles": smiles, "instruction": "add methyl"},
            reasoning="Hypothesis: small modification improves potency without breaking ADMET.",
        )
    if stage == 4:
        return DrugDiscoveryAction(
            tool="evaluate_admet",
            params={"smiles": smiles},
            reasoning="Check Lipinski/PAINS/hERG before lead validation -- safety first.",
        )
    return DrugDiscoveryAction(
        tool="validate_compound",
        params={"smiles": smiles, "panel": ["hERG", "CYP3A4", "5HT2B"]},
        reasoning="Final docking + selectivity panel before nominating the lead.",
    )


@dataclass
class RolloutSummary:
    episode: int
    disease: str
    total_reward: float
    breakdown: dict
    steps: int
    budget_remaining: float
    final_stage: int
    best_smiles: Optional[str]


def play_one(env: DrugDiscoveryEnv, max_steps: int, episode: int) -> RolloutSummary:
    obs = env.reset()
    done = False
    last_reward = 0.0
    breakdown: dict = {}
    steps = 0
    while not done and steps < max_steps:
        action = random_policy(env, obs)
        obs = env.step(action)
        last_reward = float(obs.reward or 0.0)
        if obs.reward_breakdown is not None:
            breakdown = obs.reward_breakdown.to_dict()
        done = obs.done
        steps += 1
    state = env._game_state  # noqa: SLF001
    best = state.best_compound() if state else None
    return RolloutSummary(
        episode=episode,
        disease=state.disease if state else "",
        total_reward=last_reward,
        breakdown=breakdown,
        steps=steps,
        budget_remaining=state.budget_remaining if state else 0.0,
        final_stage=state.stage if state else 0,
        best_smiles=best.smiles if best else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-policy baseline for the drug discovery env.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--out", type=str, default="outputs/baseline/baseline_summaries.json")
    parser.add_argument("--data-mode", type=str, default="local_only", choices=[m.value for m in DataSourceMode])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode(args.data_mode)
    env = DrugDiscoveryEnv(settings=settings)

    random.seed(args.seed)
    summaries = [play_one(env, args.max_steps, i + 1) for i in range(args.episodes)]
    mean = sum(s.total_reward for s in summaries) / max(1, len(summaries))
    print(f"\nBaseline mean reward over {len(summaries)} episodes: {mean:.3f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
