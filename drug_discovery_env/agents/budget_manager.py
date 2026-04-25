from __future__ import annotations

from drug_discovery_env.core.state import GameState


class BudgetManagerAgent:
    def run(self, state: GameState) -> list[str]:
        return [m["message"] for m in self.messages(state)]

    def messages(self, state: GameState) -> list[dict[str, str]]:
        remain_ratio = state.budget_remaining / max(1.0, state.budget_initial)
        if remain_ratio < 0.1:
            return [{
                "agent": "budget",
                "severity": "block",
                "message": (
                    f"Budget critically low ({state.budget_remaining:.1f}/"
                    f"{state.budget_initial:.0f}). Stop spending; nominate a lead now."
                ),
            }]
        if remain_ratio < 0.3:
            return [{
                "agent": "budget",
                "severity": "warn",
                "message": (
                    f"70%+ of budget consumed ({state.budget_remaining:.1f} left). "
                    "Prefer cheap tools (search, ADMET) over docking/validation."
                ),
            }]
        return [{
            "agent": "budget",
            "severity": "info",
            "message": "Budget healthy: maintain exploration with measured risk.",
        }]
