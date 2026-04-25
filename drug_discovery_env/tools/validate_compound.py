from __future__ import annotations

import random

from drug_discovery_env.chemistry import RDKitLab
from drug_discovery_env.core.state import GameState
from drug_discovery_env.tools.base import Tool


class ValidateCompoundTool(Tool):
    """Final validation: docking + off-target panel.

    Backed by the RDKit-based `RDKitLab.run_docking` when a target is known,
    falling back to potency-derived noise otherwise.
    """

    name = "validate_compound"

    def __init__(self, lab: RDKitLab | None = None) -> None:
        self.lab = lab or RDKitLab()

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        smiles = str(params.get("smiles", ""))
        panel = list(params.get("panel", ["hERG", "CYP3A4", "5HT2B"]))
        if smiles not in state.compound_ledger:
            raise ValueError("Unknown compound")
        rec = state.compound_ledger[smiles]
        target = (state.target or {}).get("target") if state.target else None
        if target:
            res = self.lab.run_docking(smiles, target)
            docking = max(0.0, min(1.0, res["quality"]))
        else:
            docking = max(0.0, min(1.0, rec.potency + random.gauss(0, 0.08)))
        selectivity = max(0.0, min(1.0, rec.selectivity + random.gauss(0, 0.07)))
        off_target_hits = {p: max(0.0, min(1.0, random.gauss(0.35, 0.2))) for p in panel}
        rec.selectivity = selectivity
        rec.metadata["validation"] = {
            "docking": docking,
            "panel": off_target_hits,
        }
        return {
            "smiles": smiles,
            "docking": docking,
            "selectivity": selectivity,
            "off_target_hits": off_target_hits,
        }
