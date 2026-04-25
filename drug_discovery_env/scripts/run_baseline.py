"""Random-policy baseline — true random valid action per stage.

Floor that GRPO has to beat. Uses the in-process env (not HTTP) so it works
without a running server.
"""

from __future__ import annotations

import argparse
import json
import random as _r
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction
from drug_discovery_env.server.environment import DrugDiscoveryEnv


@dataclass
class RolloutSummary:
    total_reward: float
    breakdown: Dict[str, float]
    steps: int
    budget_remaining: float
    advanced_lead: Optional[str]
    terminated_reason: Optional[str]
    final_stage: str


def random_policy(obs: Any) -> DrugDiscoveryAction:
    stage = obs.stage
    actives = obs.active_compounds or []
    target = obs.selected_target

    if stage == "target_selection":
        return DrugDiscoveryAction(
            tool="advance_stage",
            params={"target": _r.choice(["DPP4", "GLP1R", "ACE", "SERT", "EGFR", "BACE1"])},
            reasoning="Pick a plausible target.",
        )
    if not target:
        return DrugDiscoveryAction(
            tool="select_target",
            params={"target": "DPP4"},
            reasoning="Need a target before any chemistry.",
        )
    if not actives:
        return DrugDiscoveryAction(
            tool="search_compounds",
            params={"min_qed": 0.4},
            reasoning="Need a candidate pool.",
        )
    smiles = _r.choice(actives)["smiles"]
    if stage == "hit_id":
        return _r.choice([
            DrugDiscoveryAction(
                tool="predict_affinity",
                params={"smiles": smiles, "target": target},
                reasoning="Quick potency check.",
            ),
            DrugDiscoveryAction(tool="advance_stage", reasoning="Move on."),
        ])
    if stage == "hit_to_lead":
        return _r.choice([
            DrugDiscoveryAction(
                tool="modify_molecule",
                params={"smiles": smiles, "instruction": "add methyl"},
                reasoning="SAR variation.",
            ),
            DrugDiscoveryAction(tool="advance_stage", reasoning="Advance."),
        ])
    if stage == "admet":
        return _r.choice([
            DrugDiscoveryAction(
                tool="evaluate_admet",
                params={"smiles": smiles},
                reasoning="ADMET screen.",
            ),
            DrugDiscoveryAction(tool="advance_stage", reasoning="Advance."),
        ])
    if stage == "lead_validation":
        return _r.choice([
            DrugDiscoveryAction(
                tool="validate_compound",
                params={"smiles": smiles, "target": target},
                reasoning="Final validation.",
            ),
            DrugDiscoveryAction(tool="advance_stage", reasoning="Nominate lead."),
        ])
    return DrugDiscoveryAction(tool="pause_and_review_all", reasoning="Pause.")


def play_one(env: DrugDiscoveryEnv) -> RolloutSummary:
    obs = env.reset()
    total_reward = 0.0
    breakdown: Dict[str, float] = {}
    steps = 0
    while not obs.done and steps < obs.max_steps + 1:
        steps += 1
        action = random_policy(obs)
        obs = env.step(action)
        if obs.reward is not None:
            total_reward = float(obs.reward)
        if obs.last_result and "reward_breakdown" in obs.last_result:
            breakdown = obs.last_result["reward_breakdown"]
    state = env.state
    return RolloutSummary(
        total_reward=total_reward,
        breakdown=breakdown,
        steps=steps,
        budget_remaining=state.budget_remaining,
        advanced_lead=state.advanced_compound_id,
        terminated_reason=state.terminated_reason,
        final_stage=state.stage,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-policy baseline runner")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--data-mode", choices=[m.value for m in DataSourceMode], default="hybrid")
    parser.add_argument("--out-dir", type=str, default="outputs/baseline")
    args = parser.parse_args()

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode(args.data_mode)
    env = DrugDiscoveryEnv(settings=settings)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[RolloutSummary] = []
    for ep in range(args.episodes):
        s = play_one(env)
        summaries.append(s)
        print(
            f"[baseline {ep + 1}/{args.episodes}] reward={s.total_reward:.3f}  "
            f"steps={s.steps}  budget_left={s.budget_remaining:.0f}  end={s.terminated_reason}"
        )

    avg = sum(s.total_reward for s in summaries) / max(1, len(summaries))
    print(f"\nBaseline mean reward over {len(summaries)} episodes: {avg:.3f}")
    (out_dir / "baseline_summaries.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
