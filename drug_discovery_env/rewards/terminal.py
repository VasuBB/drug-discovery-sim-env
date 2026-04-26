"""Terminal compound reward — smooth, multiplicative form.

Hard floors (hERG / PAINS) zero the score outright. Otherwise the score is a
single product of saturating terms, so each factor contributes a smooth
gradient that GRPO can follow:

    pIC50 = -log10(affinity_nM / 1e9)            # ~3..10
    binding   = sigmoid(potency_weight * (pIC50 - 6))
    safety    = exp(-herg_prob)
    admet     = 0.5 + 0.5 * ro5_pass
    novelty   = novelty_weight * (1 / (1 + dup_count))
    score     = binding * safety * admet * novelty

`affinity_nM` and `docking_score` fall back to the multi-dim potency / safety
fields when the raw assay readouts are absent, so the formula is robust to
either path through the env.
"""

from __future__ import annotations

import math
from typing import List, Optional

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import CompoundRecord


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
        components = self.settings.reward.terminal_components
        admet = compound.admet or {}
        herg = float(compound.metadata.get("admet", {}).get("herg_prob", admet.get("herg_prob", 0.0)))

        if herg > float(floors["herg_prob_max"]):
            return 0.0
        if admet.get("pains"):
            return 0.0

        if compound.binding_affinity_nM is not None and compound.binding_affinity_nM > 0:
            pic50 = -math.log10(max(compound.binding_affinity_nM, 1e-3) / 1e9)
        else:
            pic50 = 3.0 + 7.0 * max(0.0, min(1.0, compound.potency))

        potency_w = float(components.get("potency_weight", 1.2))
        novelty_w = float(components.get("novelty_weight", 1.0))

        binding = _sigmoid(potency_w * (pic50 - 6.0))
        safety = math.exp(-herg) * (0.5 + 0.5 * (1.0 - float(admet.get("tox_score", 0.0))))
        ro5 = 1.0 if admet.get("ro5_pass") else 0.0
        admet_factor = 0.5 + 0.5 * ro5

        dup_count = sum(1 for s in (seen_smiles or []) if s == compound.smiles) - 1
        dup_count = max(0, dup_count)
        novelty = novelty_w / (1.0 + dup_count)

        score = binding * safety * admet_factor * novelty
        return max(0.0, min(1.0, score))
