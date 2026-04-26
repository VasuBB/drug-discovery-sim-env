"""Process reward — info gain per unit cost over the recent action window.

Single smooth form:
    score = mean( info_gain / (1 + log1p(cost_paid)) )    over last 5 actions
"""

from __future__ import annotations

import math

from drug_discovery_env.core.state import GameState

_WINDOW = 5


class ProcessReward:
    def score(self, state: GameState) -> float:
        if not state.action_history:
            return 0.0
        recent = state.action_history[-_WINDOW:]
        total = 0.0
        for record in recent:
            cost = max(0.0, float(record.cost_paid))
            gain = max(0.0, float(record.information_gain))
            total += gain / (1.0 + math.log1p(cost))
        score = total / len(recent)
        return max(0.0, min(1.0, score))
