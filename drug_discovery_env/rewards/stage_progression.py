"""Stage-progression reward — credit for clearing canonical stages without skipping."""

from __future__ import annotations

from typing import List

CANONICAL = ("target_selection", "hit_id", "hit_to_lead", "admet", "lead_validation")


class StageProgressionReward:
    def score(self, stages_completed: List[str]) -> float:
        cleared = 0
        for stage in CANONICAL:
            if stage in stages_completed:
                cleared += 1
            else:
                break
        # cap at 1.0 when all 5 cleared
        return cleared / len(CANONICAL)
