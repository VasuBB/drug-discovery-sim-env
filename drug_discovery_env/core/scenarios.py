"""Disease/target scenario seeds for the campaign.

Each scenario carries a difficulty hint that the reward shaping uses to scale
terminal reward. A real submission would pull these from ChEMBL; we ship a
small built-in library so the env runs out of the box.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Scenario:
    disease: str
    canonical_target: str
    alternative_targets: List[str] = field(default_factory=list)
    difficulty: float = 0.5

    def all_targets(self) -> List[str]:
        return [self.canonical_target] + list(self.alternative_targets)


_SCENARIOS: List[Scenario] = [
    Scenario("Type 2 Diabetes", "DPP4", ["GLP1R", "SGLT2", "PPARG"], 0.30),
    Scenario("Hypertension", "ACE", ["AGTR1", "REN", "ADRB1"], 0.25),
    Scenario("Chronic Myeloid Leukemia", "BCR-ABL", ["ABL1", "SRC", "KIT"], 0.50),
    Scenario("Alzheimer's Disease", "BACE1", ["AChE", "MAPT", "GSK3B"], 0.70),
    Scenario("HIV", "HIV-1 Protease", ["HIV-1 RT", "CCR5", "Integrase"], 0.40),
    Scenario("Rheumatoid Arthritis", "TNF-alpha", ["JAK2", "IL6R", "COX2"], 0.45),
    Scenario("Asthma", "ADRB2", ["LTD4R", "PDE4", "IL5"], 0.30),
    Scenario("Major Depressive Disorder", "SERT", ["NET", "DAT", "MAOA"], 0.50),
    Scenario("Epilepsy", "SCN1A", ["GABA-A", "CACNA1A", "GRIN2B"], 0.60),
    Scenario("Parkinson's Disease", "DRD2", ["MAO-B", "COMT", "AADC"], 0.65),
    Scenario("Migraine", "CGRP-R", ["5HT1B", "5HT1D"], 0.40),
    Scenario("Atrial Fibrillation", "KCNH2", ["SCN5A", "ADRB1"], 0.55),
    Scenario("Breast Cancer (HR+)", "ESR1", ["AR", "PIK3CA", "CDK4"], 0.55),
    Scenario("Non-small Cell Lung Cancer", "EGFR", ["ALK", "KRAS", "MET"], 0.60),
    Scenario("Multiple Sclerosis", "S1PR1", ["CD20", "DHODH"], 0.65),
]


def list_scenarios() -> List[Scenario]:
    return list(_SCENARIOS)


def sample_scenario(seed: Optional[int] = None) -> Scenario:
    rng = random.Random(seed)
    return rng.choice(_SCENARIOS)


def find_scenario(disease: str) -> Optional[Scenario]:
    for s in _SCENARIOS:
        if s.disease.lower() == disease.lower():
            return s
    return None
