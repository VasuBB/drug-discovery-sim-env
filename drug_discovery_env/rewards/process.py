from __future__ import annotations

from collections import Counter

from drug_discovery_env.core.state import GameState


class ProcessReward:
    def score(self, state: GameState) -> float:
        if not state.action_history:
            return 0.0
        recent = state.action_history[-8:]
        mean_gain = sum(x.information_gain for x in recent) / len(recent)
        cost_eff = sum(x.information_gain / max(1.0, x.cost_paid) for x in recent) / len(recent)
        stage_bonus = 0.1 if state.stage >= 3 else 0.0
        uncertainty_drop = 0.0
        if state.compound_ledger:
            uncertainty_drop = 1.0 - (
                sum(c.uncertainty for c in state.compound_ledger.values()) / len(state.compound_ledger)
            )
        tool_counts = Counter(x.tool for x in recent)
        dominant = tool_counts.most_common(1)[0][1]
        loop_penalty = max(0.0, (dominant - len(recent) * 0.45) / len(recent))
        return max(
            0.0,
            min(
                1.0,
                0.34 * mean_gain
                + 0.30 * cost_eff
                + 0.20 * uncertainty_drop
                + stage_bonus
                - 0.20 * loop_penalty,
            ),
        )
