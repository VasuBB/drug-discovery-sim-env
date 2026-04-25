from __future__ import annotations

import random

from drug_discovery_env.chemistry import RDKitLab
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class EvaluateAdmetTool(Tool):
    """ADMET evaluation backed by RDKit (Lipinski / PAINS / TOX SMARTS / hERG proxy / QED).

    Falls back to a noisy heuristic when RDKit is unavailable.
    """

    name = "evaluate_admet"

    def __init__(self, lab: RDKitLab | None = None) -> None:
        self.lab = lab or RDKitLab()

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        smiles = str(params.get("smiles", ""))
        if smiles not in state.compound_ledger:
            raise ValueError("Unknown compound")
        rec = state.compound_ledger[smiles]
        admet = self.lab.compute_admet(smiles).to_dict()
        # Cell-permeability proxy (Caco-2) -- use TPSA heuristic.
        caco2 = max(0.0, min(1.0, 1.0 - (admet["tpsa"] / 200.0)))
        admet["caco2"] = caco2
        safety = max(
            0.0,
            min(
                1.0,
                1.0 - (0.45 * admet["herg_prob"] + 0.35 * admet["tox_score"] + 0.2 * (1 - caco2)),
            ),
        )
        # Apply a small assay noise so re-running gives slightly different
        # numbers (matches the 'noisy biology' of the original heuristic).
        safety = max(0.0, min(1.0, safety + random.gauss(0, 0.05)))
        rec.safety = safety
        rec.metadata["admet"] = admet
        return {"smiles": smiles, "admet": admet, "safety": safety}
