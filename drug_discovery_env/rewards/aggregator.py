from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.models import RewardBreakdown
from drug_discovery_env.core.state import GameState
from drug_discovery_env.rewards.oversight import OversightPenalty
from drug_discovery_env.rewards.process import ProcessReward
from drug_discovery_env.rewards.reasoning import ReasoningReward
from drug_discovery_env.rewards.strategy import StrategyReward
from drug_discovery_env.rewards.terminal import TerminalReward


class RewardEngine:
    """Composable rubric.

    Components (each in [0, 1] except oversight which is in [-0.2, 0]):
        terminal           -- scientific quality of the best compound
        process            -- per-step information gain / cost efficiency
        reasoning          -- structural + scientific keyword density
        strategy           -- multi-stage progress + diversity + budget
        oversight_penalty  -- penalty for ignoring 'block' sub-agent warnings
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.terminal = TerminalReward(settings)
        self.process = ProcessReward()
        self.reasoning = ReasoningReward()
        self.strategy = StrategyReward()
        self.oversight = OversightPenalty()

    def compute(self, state: GameState, action) -> RewardBreakdown:
        w = self.settings.reward.weights
        term = self.terminal.score(state.best_compound())
        proc = self.process.score(state)
        rea = self.reasoning.score(action)
        strat = self.strategy.score(state)
        penalty = self.oversight.score(state)
        total = (
            w["terminal"] * term
            + w["process"] * proc
            + w["reasoning"] * rea
            + w["strategy"] * strat
            + penalty  # already negative & weighted
        )
        return RewardBreakdown(
            terminal=term,
            process=proc,
            reasoning=rea,
            strategy=strat,
            oversight_penalty=penalty,
            total=total,
        )
