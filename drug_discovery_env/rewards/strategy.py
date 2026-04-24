from __future__ import annotations

from drug_discovery_env.core.state import GameState


class StrategyReward:
    def score(self, state: GameState) -> float:
        stage_prog = state.stage / 5
        diversity = min(1.0, len(state.compound_ledger) / 20)
        recovery = 0.2 if any("route_failure" in str(x) for x in state.budget_ledger[-5:]) else 0.0
        budget_eff = state.budget_remaining / max(1.0, state.budget_initial)
        return max(0.0, min(1.0, 0.35 * stage_prog + 0.25 * diversity + 0.25 * budget_eff + 0.15 * recovery))
