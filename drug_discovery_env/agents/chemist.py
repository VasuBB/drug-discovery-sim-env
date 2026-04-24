from __future__ import annotations

from drug_discovery_env.core.state import GameState


class ChemistAgent:
    def run(self, state: GameState) -> list[str]:
        if not state.compound_ledger:
            return ["Need initial hits before SAR optimization advice."]
        best = state.best_compound()
        if not best:
            return ["No best candidate available yet."]
        msgs = []
        if best.developability < 0.5:
            msgs.append("Improve developability by reducing lipophilicity and adding balanced polarity.")
        if best.selectivity < 0.5:
            msgs.append("Run scaffold-hopping to improve selectivity against off-target neighbors.")
        if not msgs:
            msgs.append("Promote best candidate to validation panel with orthogonal assay selection.")
        return msgs
