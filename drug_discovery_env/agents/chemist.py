"""Chemist sub-agent — diversity + SAR + post-modification re-screen prompts."""

from __future__ import annotations

from typing import Dict, List

from drug_discovery_env.core.state import GameState


def _msg(severity: str, message: str) -> Dict[str, str]:
    return {"agent": "chemist", "severity": severity, "message": message}


class ChemistAgent:
    def run(self, state: GameState) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        active = list(state.compound_ledger.values())
        if not active:
            return [_msg("info", "Need initial hits before SAR optimization advice.")]

        smiles_list = [c.smiles for c in active]
        unique = len(set(smiles_list))
        if unique < max(1, len(smiles_list) // 2):
            out.append(_msg(
                "warn",
                "Active pool has low structural diversity; consider modifying scaffolds before nominating a lead.",
            ))

        if state.last_tool == "modify_molecule":
            out.append(_msg(
                "info",
                "Modification recorded. Re-screen the variant for binding and ADMET before advancing.",
            ))

        if state.last_tool == "predict_affinity":
            score = (state.last_result or {}).get("score")
            if isinstance(score, (int, float)) and score > 0.7:
                out.append(_msg(
                    "info",
                    f"Strong predicted potency (score {score:.2f}); consider scaffold-hopping for selectivity.",
                ))

        best = state.best_compound()
        if best is not None:
            if best.developability < 0.5:
                out.append(_msg(
                    "info",
                    "Best candidate has weak developability — try lipophilicity reduction or polarity tuning.",
                ))
            if best.selectivity < 0.5 and state.stage in {"hit_to_lead", "lead_validation"}:
                out.append(_msg(
                    "warn",
                    "Best candidate selectivity is borderline — scaffold-hop or run wider validation panel.",
                ))
        return out
