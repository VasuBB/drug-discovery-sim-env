"""Group-reward helper used by GRPO when sampling from offline rollouts."""

from __future__ import annotations

from drug_discovery_env.training.rollout_generator import generate_rollouts


def compute_group_rewards(group_size: int) -> list[float]:
    samples = generate_rollouts(max(1, group_size // 2))
    rewards = [s.reward for s in samples]
    if not rewards:
        return [0.0] * group_size
    rewards = rewards[:group_size]
    while len(rewards) < group_size:
        rewards.append(rewards[-1])
    return rewards
