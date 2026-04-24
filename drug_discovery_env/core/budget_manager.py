from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import ActionRecord, GameState


class BudgetManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def compute_tool_cost(self, tool_name: str, information_gain: float, uncertainty: float, stage: int) -> float:
        costs = self.settings.tools.costs
        multipliers = self.settings.budget.variable_cost_multipliers
        base = float(costs.get(tool_name, 0.0))
        low_info_factor = multipliers["low_information"] if information_gain < 0.3 else 1.0
        high_uncertainty_factor = multipliers["high_uncertainty"] if uncertainty > 0.6 else 1.0
        topology_factor = multipliers["topology_complexity"] if stage >= 4 else 1.0
        opportunity = 1.0 + self.settings.budget.opportunity_cost_factor
        redundancy = (
            1.0 + self.settings.budget.late_stage_redundancy_penalty
            if stage >= 4 and information_gain < 0.25
            else 1.0
        )
        return base * low_info_factor * high_uncertainty_factor * topology_factor * opportunity * redundancy

    def pay(self, state: GameState, tool_name: str, information_gain: float, uncertainty: float) -> float:
        cost = self.compute_tool_cost(tool_name, information_gain, uncertainty, state.stage)
        state.add_budget_event(reason=f"tool:{tool_name}", amount=cost)
        state.action_history.append(
            ActionRecord(
                step=state.step,
                tool=tool_name,
                params={},
                information_gain=information_gain,
                cost_paid=cost,
            )
        )
        return cost
