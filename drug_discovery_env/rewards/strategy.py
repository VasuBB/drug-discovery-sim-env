from __future__ import annotations

from drug_discovery_env.core.state import GameState


class StrategyReward:
    def score(self, state: GameState) -> float:
        stage_prog = state.stage / 5
        diversity = min(1.0, len(state.compound_ledger) / 20)
        recovery = 0.2 if any("route_failure" in str(x) for x in state.budget_ledger[-5:]) else 0.0
        budget_eff = state.budget_remaining / max(1.0, state.budget_initial)
        evidence_density = min(1.0, len(state.evidence_ledger) / max(1, state.step))
        risk_load = min(1.0, (sum(state.off_target_risk_profile.values()) / max(1, len(state.off_target_risk_profile))))
        return max(
            0.0,
            min(
                1.0,
                0.30 * stage_prog
                + 0.22 * diversity
                + 0.18 * budget_eff
                + 0.15 * evidence_density
                + 0.15 * recovery
                - 0.15 * risk_load,
            ),
        )
