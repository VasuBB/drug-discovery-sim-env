"""Terminal compound reward — weighted sum + hard hERG/PAINS safety floor."""

from __future__ import annotations

import math
from typing import List, Optional

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import CompoundRecord


class TerminalReward:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def score(
        self,
        compound: Optional[CompoundRecord],
        seen_smiles: Optional[List[str]] = None,
    ) -> float:
        if compound is None:
            return 0.0

        floors = self.settings.reward.floors
        admet = compound.admet or {}
        herg = float(compound.metadata.get("admet", {}).get("herg_prob", 0.0))
        if herg > float(floors["herg_prob_max"]):
            return 0.0  # hard safety floor — toxic compound can't win
        if admet.get("pains"):
            return 0.0
        if admet.get("tox_flag") and herg > 0.0:
            return 0.0

        # Multi-dimensional weighted sum (akshat)
        components = self.settings.reward.terminal_components
        values = {
            "potency": compound.potency,
            "selectivity": compound.selectivity,
            "safety": compound.safety,
            "synthesizability": compound.synthesizability,
            "novelty": compound.novelty,
            "developability": compound.developability,
        }
        weighted = sum(float(components[k]) * max(0.0, min(1.0, values[k])) for k in components)

        # Augment with main's affinity / docking / ADMET-flag credit when present
        bonus = 0.0
        if compound.binding_affinity_nM is not None:
            bonus += 0.10 * max(0.0, 1.0 - math.log10(max(compound.binding_affinity_nM, 0.1)) / 4.0)
        if compound.docking_score is not None:
            clamped = max(-12.0, min(-4.0, compound.docking_score))
            bonus += 0.10 * ((-clamped - 4.0) / 8.0)
        if admet:
            admet_term = 0.0
            if admet.get("ro5_pass"):
                admet_term += 0.4
            if not admet.get("pains", False):
                admet_term += 0.3
            if not admet.get("tox_flag", False):
                admet_term += 0.3
            bonus += 0.10 * admet_term

        # Novelty discount if repeated against an existing pool
        if seen_smiles and compound.smiles in seen_smiles[:5]:
            bonus *= 0.4

        return max(0.0, min(1.0, weighted + bonus))
