"""Stage progression reward — fraction of canonical stages cleared.

    score = cleared / 4    (4 == number of advance_stage transitions)
"""

from __future__ import annotations

from typing import Iterable


class StageProgressionReward:
    def score(self, stages_completed: Iterable[str]) -> float:
        n = sum(1 for _ in stages_completed)
        return max(0.0, min(1.0, n / 4.0))
