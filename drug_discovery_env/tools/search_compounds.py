"""search_compounds — populate the compound pool from the live provider."""

from __future__ import annotations

from typing import Any, Dict

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class SearchCompoundsTool(Tool):
    name = "search_compounds"
    default_information_gain = 0.5

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        min_qed = float(params.get("min_qed", 0.4))

        try:
            compounds = self.provider.get_compounds({"min_qed": min_qed, "target": state.target})
        except Exception as exc:
            return {
                "error": "compound_lookup_failed",
                "message": str(exc),
                "hits": [],
                "added": [],
                "count": 0,
                "source": "live",
                "confidence": 0.0,
            }

        added = []
        for c in compounds[:20]:
            smiles = c.get("smiles", "")
            if not smiles:
                continue
            rec = state.add_compound(
                smiles,
                origin=str(c.get("source", "search")),
                developability=float(c.get("qed", 0.0)),
                uncertainty=0.8,
            )
            added.append({"id": rec.id, "smiles": smiles, "qed": c.get("qed")})
        return {
            "hits": compounds[:20],
            "added": added,
            "count": len(added),
            "source": "live",
            "confidence": max((float(c.get("confidence", 0.0)) for c in compounds[:20]), default=0.0),
        }
