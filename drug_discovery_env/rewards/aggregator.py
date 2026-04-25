"""Reward engine — composes the 7-component rubric and emits a RewardBreakdown.

Components:
  terminal_compound      multi-dim weighted sum + hard hERG/PAINS floor
  stage_progression      cleared canonical stages without skipping
  budget_efficiency      budget remaining / total
  reasoning_depth        keyword + length scoring over reasoning traces
  process                per-step info-gain / cost efficiency
  strategy               compound diversity + setback recovery
  oversight_penalty      penalty for ignoring `block` warnings (subtracted)
"""

from __future__ import annotations

from typing import List, Optional

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.models import DrugDiscoveryAction, RewardBreakdown
from drug_discovery_env.core.state import GameState
from drug_discovery_env.rewards.budget_efficiency import BudgetEfficiencyReward
from drug_discovery_env.rewards.oversight import OversightPenalty
from drug_discovery_env.rewards.process import ProcessReward
from drug_discovery_env.rewards.reasoning import ReasoningReward
from drug_discovery_env.rewards.stage_progression import StageProgressionReward
from drug_discovery_env.rewards.strategy import StrategyReward
from drug_discovery_env.rewards.terminal import TerminalReward


class RewardEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.terminal = TerminalReward(settings)
        self.stage_progression = StageProgressionReward()
        self.budget_efficiency = BudgetEfficiencyReward()
        self.reasoning = ReasoningReward()
        self.process = ProcessReward()
        self.strategy = StrategyReward()
        self.oversight = OversightPenalty()

    def compute(
        self,
        state: GameState,
        action: Optional[DrugDiscoveryAction] = None,
        *,
        terminal: bool = False,
    ) -> RewardBreakdown:
        weights = self.settings.reward.weights

        # advanced compound for terminal scoring
        nominated = state.get_by_id(state.advanced_compound_id) if state.advanced_compound_id else None
        if nominated is None and terminal:
            nominated = state.best_compound()

        seen: List[str] = [c.smiles for c in state.compound_ledger.values()]

        terminal_v = self.terminal.score(nominated, seen_smiles=seen) if terminal else 0.0
        stage_v = self.stage_progression.score(state.stages_completed)
        budget_v = self.budget_efficiency.score(state.budget_remaining, state.budget_initial)
        reasoning_v = self.reasoning.score(state.reasoning_traces)
        process_v = self.process.score(state)
        strategy_v = self.strategy.score(state)
        oversight_v = self.oversight.score(state.warnings_issued, state.warnings_ignored)

        total = (
            weights["terminal"] * terminal_v
            + weights["stage_progression"] * stage_v
            + weights["budget_efficiency"] * budget_v
            + weights["reasoning_depth"] * reasoning_v
            + weights["process"] * process_v
            + weights["strategy"] * strategy_v
            - weights["oversight_penalty"] * oversight_v
        )

        return RewardBreakdown(
            terminal_compound=terminal_v,
            stage_progression=stage_v,
            budget_efficiency=budget_v,
            reasoning_depth=reasoning_v,
            process=process_v,
            strategy=strategy_v,
            oversight_penalty=-oversight_v,  # surface as negative for clarity
            total=max(0.0, min(1.0, total)),
        )
