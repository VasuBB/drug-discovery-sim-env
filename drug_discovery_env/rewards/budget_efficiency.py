"""Budget-efficiency reward — credit for budget remaining at termination."""

from __future__ import annotations


class BudgetEfficiencyReward:
    def score(self, budget_remaining: float, budget_total: float) -> float:
        if budget_total <= 0:
            return 0.0
        return max(0.0, min(1.0, budget_remaining / budget_total))
