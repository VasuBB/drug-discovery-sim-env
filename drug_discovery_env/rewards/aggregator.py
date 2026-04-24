from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.models import RewardBreakdown
from drug_discovery_env.core.state import GameState
from drug_discovery_env.rewards.process import ProcessReward
from drug_discovery_env.rewards.reasoning import ReasoningReward
from drug_discovery_env.rewards.strategy import StrategyReward
from drug_discovery_env.rewards.terminal import TerminalReward


class RewardEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.terminal = TerminalReward(settings)
        self.process = ProcessReward()
        self.reasoning = ReasoningReward()
        self.strategy = StrategyReward()

    def compute(self, state: GameState, action) -> RewardBreakdown:
        w = self.settings.reward.weights
        term = self.terminal.score(state.best_compound())
        proc = self.process.score(state)
        rea = self.reasoning.score(action)
        strat = self.strategy.score(state)
        total = (
            w["terminal"] * term
            + w["process"] * proc
            + w["reasoning"] * rea
            + w["strategy"] * strat
        )
        return RewardBreakdown(terminal=term, process=proc, reasoning=rea, strategy=strat, total=total)
