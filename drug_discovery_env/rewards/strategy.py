"""Strategy reward — saturating diversity blended with stage progress.

    diversity = 1 - exp(-num_compounds / 3)
    stage     = stage_index / (len(STAGE_ORDER) - 1)
    score     = (diversity + stage) / 2
"""

from __future__ import annotations

import math

from drug_discovery_env.core.models import STAGE_ORDER
from drug_discovery_env.core.state import GameState


class StrategyReward:
    def score(self, state: GameState) -> float:
        diversity = 1.0 - math.exp(-len(state.compound_ledger) / 3.0)
        denom = max(1, len(STAGE_ORDER) - 1)
        stage_idx = STAGE_ORDER.index(state.stage) if state.stage in STAGE_ORDER else 0
        stage_progress = stage_idx / denom
        return max(0.0, min(1.0, 0.5 * diversity + 0.5 * stage_progress))
