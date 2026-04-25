"""Process reward — per-step info-gain / cost efficiency."""

from __future__ import annotations

from drug_discovery_env.core.state import GameState


class ProcessReward:
    def score(self, state: GameState) -> float:
        if not state.action_history:
            return 0.0
        recent = state.action_history[-5:]
        mean_gain = sum(x.information_gain for x in recent) / len(recent)
        cost_eff = sum(x.information_gain / max(1.0, x.cost_paid) for x in recent) / len(recent)
        # promote getting past hit_id (akshat hint)
        stage_bonus = 0.1 if state.stage in {"hit_to_lead", "admet", "lead_validation", "finished"} else 0.0
        uncertainty_drop = 0.0
        if state.compound_ledger:
            uncertainty_drop = 1.0 - (
                sum(c.uncertainty for c in state.compound_ledger.values()) / len(state.compound_ledger)
            )
        return max(0.0, min(1.0, 0.4 * mean_gain + 0.35 * cost_eff + 0.15 * uncertainty_drop + stage_bonus))
