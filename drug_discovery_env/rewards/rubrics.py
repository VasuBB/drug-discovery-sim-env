from __future__ import annotations

from typing import Any

try:
    from openenv.core.rubrics import Rubric, WeightedSum
except Exception:  # pragma: no cover
    from openenv_core.rubrics import Rubric, WeightedSum


class BreakdownFieldRubric(Rubric):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def forward(self, action: Any, observation: Any) -> float:
        _ = action
        rb = getattr(observation, "reward_breakdown", None)
        if rb is None:
            return 0.0
        return float(getattr(rb, self.field, 0.0))


class SafetyGateRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        _ = action
        rb = getattr(observation, "reward_breakdown", None)
        if rb is None:
            return 0.0
        # If terminal quality collapses, strongly suppress reward.
        return 1.0 if float(getattr(rb, "terminal", 0.0)) > 0.05 else 0.0


class RubricRewardComposer:
    def __init__(self, weights: dict[str, float]) -> None:
        self.terminal = BreakdownFieldRubric("terminal")
        self.process = BreakdownFieldRubric("process")
        self.reasoning = BreakdownFieldRubric("reasoning")
        self.strategy = BreakdownFieldRubric("strategy")
        self.safety_gate = SafetyGateRubric()
        self.weighted = WeightedSum(
            rubrics=[self.terminal, self.process, self.reasoning, self.strategy],
            weights=[weights["terminal"], weights["process"], weights["reasoning"], weights["strategy"]],
        )

    def score(self, action: Any, observation: Any) -> float:
        gate = self.safety_gate(action, observation)
        return float(gate * self.weighted(action, observation))
