"""Oversight sub-agent — flags policy violations and low-info loops."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


def _msg(severity: str, message: str) -> Dict[str, str]:
    return {"agent": "oversight", "severity": severity, "message": message}


class OversightAgent:
    """Two complementary detectors:

    1. Block-warning compliance: was the most recent agent action taken in
       defiance of an open `block` warning from another sub-agent?
    2. Low-info loop detection: is the agent burning steps on the same
       low-information tool over and over?
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: GameState) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []

        # 1) Compliance: did the last action violate an open block warning?
        block_msgs = [w for w in state.open_block_warnings if w.get("severity") == "block"]
        if block_msgs:
            if state.last_tool in ("advance_stage", "validate_compound") and any(
                w["agent"] == "toxicologist" for w in block_msgs
            ):
                out.append(_msg(
                    "block",
                    "Project Lead acted on a compound despite an active toxicology BLOCK warning.",
                ))
            if state.last_tool == "validate_compound" and any(w["agent"] == "budget" for w in block_msgs):
                out.append(_msg(
                    "block",
                    "Project Lead ran high-cost validation despite an active budget BLOCK warning.",
                ))

        # 2) Low-info loop detection over a sliding window
        window = self.settings.agents.oversight_loop_window
        low_info_threshold = self.settings.agents.oversight_low_info_threshold
        recent = state.action_history[-window:]
        if recent:
            tool_counts = Counter(x.tool for x in recent)
            dominant_tool, dominant_n = tool_counts.most_common(1)[0]
            mean_info = sum(x.information_gain for x in recent) / len(recent)
            if dominant_n >= max(4, window // 2) and mean_info < low_info_threshold:
                out.append(_msg(
                    "warn",
                    f"Low-information loop detected around {dominant_tool}; diversify experiments.",
                ))
            if state.stage in {"admet", "lead_validation"} and mean_info < low_info_threshold:
                out.append(_msg(
                    "warn",
                    "Unsafe acceleration in late stage with weak evidence density.",
                ))
        return out
