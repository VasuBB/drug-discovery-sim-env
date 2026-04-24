from __future__ import annotations

import random

from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class EvaluateAdmetTool(Tool):
    name = "evaluate_admet"

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        smiles = str(params.get("smiles", ""))
        if smiles not in state.compound_ledger:
            raise ValueError("Unknown compound")
        rec = state.compound_ledger[smiles]
        herg_prob = max(0.0, min(1.0, 0.4 + random.gauss(0, 0.12) + (1 - rec.potency) * 0.2))
        caco2 = max(0.0, min(1.0, 0.5 + random.gauss(0, 0.1)))
        tox = max(0.0, min(1.0, 0.35 + random.gauss(0, 0.15)))
        safety = max(0.0, min(1.0, 1.0 - (0.45 * herg_prob + 0.35 * tox + 0.2 * (1 - caco2))))
        rec.safety = safety
        rec.metadata["admet"] = {"herg_prob": herg_prob, "caco2": caco2, "tox": tox}
        return {"smiles": smiles, "admet": rec.metadata["admet"], "safety": safety}
