"""Budget Manager sub-agent — 70% warn / 90% block thresholds."""

from __future__ import annotations

from typing import Dict, List

from drug_discovery_env.core.state import GameState


def _msg(severity: str, message: str) -> Dict[str, str]:
    return {"agent": "budget", "severity": severity, "message": message}


class BudgetManagerAgent:
    def run(self, state: GameState) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if state.budget_initial <= 0:
            return out
        frac_used = 1.0 - (state.budget_remaining / state.budget_initial)
        if frac_used >= 0.9:
            out.append(_msg(
                "block",
                f"Budget critically low ({state.budget_remaining:.1f} of {state.budget_initial:.0f} left). "
                "Stop spending; nominate a lead now.",
            ))
        elif frac_used >= 0.7:
            out.append(_msg(
                "warn",
                f"70%+ of budget consumed ({state.budget_remaining:.1f} left). "
                "Prefer cheap tools (search, ADMET) over docking.",
            ))
        if state.last_cost >= 50:
            out.append(_msg(
                "info",
                f"Last action cost {state.last_cost:.0f} units — high-cost tool; reserve for verified candidates.",
            ))
        return out
