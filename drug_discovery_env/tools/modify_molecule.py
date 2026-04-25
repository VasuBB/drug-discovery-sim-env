from __future__ import annotations

import random

from drug_discovery_env.chemistry import RDKitLab
from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import CompoundRecord, GameState
from drug_discovery_env.tools.base import Tool


class ModifyMoleculeTool(Tool):
    name = "modify_molecule"

    def __init__(self, settings: Settings, lab: RDKitLab | None = None) -> None:
        self.settings = settings
        self.lab = lab or RDKitLab()

    def execute(self, state: GameState, params: dict[str, object]) -> dict[str, object]:
        smiles = str(params.get("smiles", ""))
        strategy = str(params.get("strategy", "polarity_tune"))
        instruction = str(params.get("instruction", strategy))
        if smiles not in state.compound_ledger:
            raise ValueError("Unknown compound")

        rec = state.compound_ledger[smiles]
        new_smiles = self.lab.modify_molecule(smiles, instruction)["smiles"]
        if new_smiles == smiles:
            new_smiles = f"{smiles}.M{random.randint(1, 9)}"
        new_rec = CompoundRecord(
            smiles=new_smiles,
            potency=max(0.0, min(1.0, rec.potency + random.uniform(-0.05, 0.12))),
            selectivity=max(0.0, min(1.0, rec.selectivity + random.uniform(0.01, 0.15))),
            safety=max(0.0, min(1.0, rec.safety + random.uniform(-0.08, 0.1))),
            synthesizability=max(0.0, min(1.0, rec.synthesizability + random.uniform(-0.03, 0.08))),
            novelty=max(0.0, min(1.0, rec.novelty + random.uniform(0.03, 0.12))),
            developability=max(0.0, min(1.0, rec.developability + random.uniform(0.01, 0.08))),
            uncertainty=max(0.05, rec.uncertainty - 0.05),
            metadata={"parent": smiles, "strategy": strategy},
        )
        state.compound_ledger[new_smiles] = new_rec
        return {"parent": smiles, "child": new_smiles, "strategy": strategy}
