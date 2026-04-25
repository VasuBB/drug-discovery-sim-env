"""Oversight-compliance reward (negative penalty).

Penalises the Project Lead for ignoring 'block'-severity sub-agent warnings
that were already on the table at the moment it took its action. Coupled
with the severity-aware sub-agents in `drug_discovery_env.agents`.
"""

from __future__ import annotations

from drug_discovery_env.core.state import GameState


class OversightPenalty:
    """Score is in [-0.2, 0]. Returned as raw signal -- aggregator weights it."""

    MAX_PENALTY = -0.2

    def score(self, state: GameState) -> float:
        warnings_issued = int(state.uncertainty_estimates.get("warnings_issued", 0))
        warnings_ignored = int(state.uncertainty_estimates.get("warnings_ignored", 0))
        if warnings_issued <= 0:
            return 0.0
        ignored_frac = max(0.0, min(1.0, warnings_ignored / warnings_issued))
        return self.MAX_PENALTY * ignored_frac
