from __future__ import annotations

from drug_discovery_env.core.state import GameState


class BudgetManagerAgent:
    def run(self, state: GameState) -> list[str]:
        remain_ratio = state.budget_remaining / max(1.0, state.budget_initial)
        if remain_ratio < 0.2:
            return ["Critical budget: prioritize high-information assays and stop low-yield iterations."]
        if remain_ratio < 0.4:
            return ["Budget warning: switch to selective experiments with better expected ROI."]
        return ["Budget healthy: maintain exploration with measured risk."]
