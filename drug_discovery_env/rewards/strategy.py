"""Strategy reward — diversity + recovery-from-failure.

Note: budget efficiency moved to its own component, so this score is now
purely about *what* the agent built (diversity) and *how it handled setbacks*
(recovery), not how much it spent.
"""

from __future__ import annotations

from drug_discovery_env.core.models import STAGE_ORDER
from drug_discovery_env.core.state import GameState


class StrategyReward:
    def score(self, state: GameState) -> float:
        stage_idx = STAGE_ORDER.index(state.stage) if state.stage in STAGE_ORDER else 0
        stage_prog = stage_idx / max(1, len(STAGE_ORDER) - 1)
        diversity = min(1.0, len(state.compound_ledger) / 20)
        recovery = 0.0
        # if a recent budget event mentions a route_failure and the agent kept
        # making progress (compounds added) afterwards → recovery credit
        if any("route_failure" in str(x) for x in state.budget_ledger[-5:]):
            if state.compound_ledger:
                recovery = 0.3
        return max(0.0, min(1.0, 0.5 * stage_prog + 0.4 * diversity + 0.1 * recovery))
