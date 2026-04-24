from __future__ import annotations

from collections import Counter

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


class OversightAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: GameState) -> list[str]:
        window = self.settings.agents.oversight_loop_window
        low_info_threshold = self.settings.agents.oversight_low_info_threshold
        recent = state.action_history[-window:]
        if not recent:
            return []
        tool_counts = Counter(x.tool for x in recent)
        dominant_tool, dominant_n = tool_counts.most_common(1)[0]
        mean_info = sum(x.information_gain for x in recent) / len(recent)
        messages: list[str] = []
        if dominant_n >= max(4, window // 2) and mean_info < low_info_threshold:
            messages.append(f"Low-information loop detected around {dominant_tool}; diversify experiments.")
        if state.stage >= 4 and mean_info < low_info_threshold:
            messages.append("Unsafe acceleration in late stage with weak evidence density.")
        return messages
