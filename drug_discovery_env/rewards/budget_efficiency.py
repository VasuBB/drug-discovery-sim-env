"""Budget efficiency reward — concave usage curve.

    score = (remaining / total) ** 0.7

Concavity (exponent < 1) means saving the *first* credit is worth a lot more
than saving the *last* one, so the agent isn't rewarded for hoarding budget
at the expense of campaign progress.
"""

from __future__ import annotations


class BudgetEfficiencyReward:
    def score(self, remaining: float, total: float) -> float:
        if total <= 0:
            return 0.0
        ratio = max(0.0, min(1.0, remaining / total))
        return ratio ** 0.7
