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
        s = self.settings.reward.shaping
        term = self.terminal.score(state.best_compound())
        proc = self.process.score(state)
        rea = self.reasoning.score(action)
        strat = self.strategy.score(state)

        # Potential-based shaping: reward incremental quality improvements while
        # preserving the task objective.
        potential_now = 0.45 * term + 0.25 * proc + 0.20 * strat + 0.10 * rea
        potential_delta = potential_now - state.last_potential_score
        state.last_potential_score = potential_now

        recent_tools = [x.tool for x in state.action_history[-6:]]
        loop_penalty = 0.0
        if recent_tools and recent_tools.count(recent_tools[-1]) >= 4:
            loop_penalty = float(s["loop_penalty"])

        novelty_bonus = float(s["novelty_pressure"]) * min(
            1.0,
            len(state.compound_ledger) / max(1, state.step),
        )
        unsafe_shortcut_penalty = 0.0
        if state.stage >= 4 and action.tool == "validate_compound":
            best = state.best_compound()
            best_safety = best.safety if best else 0.0
            if best_safety < 0.5:
                unsafe_shortcut_penalty = float(s["unsafe_shortcut_penalty"])

        total_core = (
            w["terminal"] * term
            + w["process"] * proc
            + w["reasoning"] * rea
            + w["strategy"] * strat
        )
        total = (
            total_core
            + float(s["potential_delta_scale"]) * potential_delta
            + novelty_bonus
            - loop_penalty
            - unsafe_shortcut_penalty
        )
        total = max(0.0, min(1.0, total))
        return RewardBreakdown(terminal=term, process=proc, reasoning=rea, strategy=strat, total=total)
