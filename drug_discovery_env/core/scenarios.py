"""Disease/target scenario seeds for the drug discovery campaign.

Ported from the `main` branch's `scenario_generator`. Every scenario carries
a difficulty hint and the canonical/alternative protein targets so the
Project Lead has a non-trivial decision in the `target_selection` stage.
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
    pathway_graph: dict[str, List[str]] = field(default_factory=dict)
    disease_nodes: List[str] = field(default_factory=list)

    def all_targets(self) -> List[str]:
        return [self.canonical_target] + list(self.alternative_targets)


_SCENARIOS: List[Scenario] = [
    Scenario(
        "Type 2 Diabetes",
        "DPP4",
        ["GLP1R", "SGLT2", "PPARG", "INSR"],
        0.30,
        {"INSR": ["PI3K", "AKT1"], "PI3K": ["MTOR", "AKT1"], "AKT1": ["MTOR", "FOXO3"]},
        ["INSR", "PI3K", "AKT1"],
    ),
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

# Canonical 5-stage progression used by the Project Lead.
STAGE_NAMES = (
    "target_selection",
    "hit_id",
    "hit_to_lead",
    "admet",
    "lead_validation",
    "finished",
)


def stage_index_to_name(stage: int) -> str:
    """Map the integer stage (1..5) to its canonical name."""

    if stage <= 0:
        return STAGE_NAMES[0]
    if stage >= len(STAGE_NAMES):
        return STAGE_NAMES[-1]
    return STAGE_NAMES[stage - 1]


def stage_name_to_index(name: str) -> int:
    try:
        return STAGE_NAMES.index(name) + 1
    except ValueError:
        return 1


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
