from __future__ import annotations

from drug_discovery_env.core.state import GameState


class ChemistAgent:
    def run(self, state: GameState) -> list[str]:
        return [m["message"] for m in self.messages(state)]

    def messages(self, state: GameState) -> list[dict[str, str]]:
        if not state.compound_ledger:
            return [{
                "agent": "chemist",
                "severity": "info",
                "message": "Need initial hits before SAR optimization advice.",
            }]
        best = state.best_compound()
        if not best:
            return [{
                "agent": "chemist",
                "severity": "info",
                "message": "No best candidate available yet.",
            }]
        out: list[dict[str, str]] = []
        smiles_list = [c.smiles for c in state.compound_ledger.values()]
        if smiles_list and len(set(smiles_list)) < max(1, len(smiles_list) // 2):
            out.append({
                "agent": "chemist",
                "severity": "warn",
                "message": (
                    "Active pool has low structural diversity; consider scaffold-hopping "
                    "before nominating a lead."
                ),
            })
        if best.developability < 0.5:
            out.append({
                "agent": "chemist",
                "severity": "info",
                "message": "Improve developability by reducing lipophilicity and adding balanced polarity.",
            })
        if best.selectivity < 0.5:
            out.append({
                "agent": "chemist",
                "severity": "info",
                "message": "Run scaffold-hopping to improve selectivity against off-target neighbors.",
            })
        if not out:
            out.append({
                "agent": "chemist",
                "severity": "info",
                "message": "Promote best candidate to validation panel with orthogonal assay selection.",
            })
        return out
