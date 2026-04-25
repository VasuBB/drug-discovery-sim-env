"""Simulated cheminformatics lab.

Wraps RDKit (where available) to produce plausible scientific feedback for
binding affinity, ADMET, structural modification, and docking. Nothing here
is real chemistry; the goal is to give the trained agent a consistent,
non-trivially-gameable reward landscape.

If RDKit is not installed, the module falls back to deterministic-pseudo-random
scores keyed off the SMILES hash so the environment still runs (useful for CI
on machines without RDKit wheels).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


def _smiles_seed(smiles: str, salt: str = "") -> int:
    h = hashlib.sha256(f"{salt}|{smiles}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _rng_for(smiles: str, salt: str = "") -> random.Random:
    return random.Random(_smiles_seed(smiles, salt))


PAINS_SMARTS = [
    "[OX1]=[CX3][CX3]=[CX3][OX1]",
    "c1ccc2c(c1)C(=O)c1ccccc1C2=O",
    "[CX3](=O)[CX3](=O)",
]

TOX_SMARTS = [
    "[Cl,Br,I][CX4]",
    "C(=O)Cl",
    "[NX3](=O)=O",
    "S(=O)(=O)Cl",
]


@dataclass
class ADMETResult:
    ro5_pass: bool
    pains: bool
    tox_flag: bool
    tox_score: float
    logp: float
    mw: float
    hbd: int
    hba: int
    tpsa: float
    qed: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ro5_pass": bool(self.ro5_pass),
            "pains": bool(self.pains),
            "tox_flag": bool(self.tox_flag),
            "tox_score": float(self.tox_score),
            "logp": float(self.logp),
            "mw": float(self.mw),
            "hbd": int(self.hbd),
            "hba": int(self.hba),
            "tpsa": float(self.tpsa),
            "qed": float(self.qed),
        }


class LabSimulator:
    COSTS = {
        "search_chembl": 5.0,
        "predict_binding_affinity": 20.0,
        "compute_admet": 10.0,
        "modify_molecule": 15.0,
        "run_docking": 80.0,
        "literature_search": 5.0,
    }

    def __init__(self, compound_library: Optional[List[str]] = None) -> None:
        self.compound_library: List[str] = compound_library or _DEFAULT_LIBRARY[:]

    def search_chembl(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        rng = _rng_for(query, "search")
        sample = rng.sample(self.compound_library, k=min(max_results, len(self.compound_library)))
        return [{"smiles": s, "source": "chembl_sim"} for s in sample]

    def literature_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        rng = _rng_for(query, "lit")
        templates = [
            "Recent studies suggest {q} pathways are druggable via small-molecule inhibition.",
            "A 2023 review of {q} highlights ADMET liabilities of first-generation leads.",
            "Crystal structures of {q} reveal an allosteric pocket adjacent to the active site.",
            "Failed Phase II trials targeting {q} were attributed to off-target hERG binding.",
            "Computational screens for {q} identified scaffolds with sub-micromolar affinity.",
            "Selectivity over related family members remains the core challenge for {q}.",
        ]
        rng.shuffle(templates)
        return [
            {"title": f"Note #{i+1} on {query}", "abstract": t.format(q=query)}
            for i, t in enumerate(templates[:max_results])
        ]

    def predict_binding_affinity(self, smiles: str, target: str) -> Dict[str, float]:
        rng = _rng_for(smiles, f"affinity:{target}")
        base = rng.gauss(2.0, 1.0)
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                mw_pen = 0.5 * abs(mw - 350) / 350
                logp_pen = 0.4 * abs(logp - 2.5) / 2.5
                base += mw_pen + logp_pen
        affinity_nM = max(0.1, 10 ** base)
        score = max(0.0, min(1.0, 1.0 - math.log10(affinity_nM) / 4.0))
        return {"affinity_nM": float(affinity_nM), "score": float(score), "target": target}

    def compute_admet(self, smiles: str) -> ADMETResult:
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._fallback_admet(smiles, invalid=True)
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            try:
                qed = Descriptors.qed(mol)
            except Exception:
                qed = 0.5
            ro5_pass = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
            pains = False
            for s in PAINS_SMARTS:
                pat = Chem.MolFromSmarts(s)
                if pat is not None and mol.HasSubstructMatch(pat):
                    pains = True
                    break
            tox_hits = 0
            for s in TOX_SMARTS:
                pat = Chem.MolFromSmarts(s)
                if pat is not None and mol.HasSubstructMatch(pat):
                    tox_hits += 1
            tox_score = min(1.0, tox_hits / 3.0)
            tox_flag = tox_score >= 0.34
            return ADMETResult(
                ro5_pass=ro5_pass,
                pains=pains,
                tox_flag=tox_flag,
                tox_score=tox_score,
                logp=logp,
                mw=mw,
                hbd=hbd,
                hba=hba,
                tpsa=tpsa,
                qed=qed,
            )
        return self._fallback_admet(smiles)

    def _fallback_admet(self, smiles: str, invalid: bool = False) -> ADMETResult:
        rng = _rng_for(smiles, "admet")
        if invalid:
            return ADMETResult(False, True, True, 1.0, 6.0, 600, 8, 12, 200, 0.05)
        return ADMETResult(
            ro5_pass=rng.random() > 0.4,
            pains=rng.random() < 0.15,
            tox_flag=rng.random() < 0.25,
            tox_score=rng.random(),
            logp=rng.uniform(-1, 6),
            mw=rng.uniform(180, 600),
            hbd=rng.randint(0, 7),
            hba=rng.randint(1, 12),
            tpsa=rng.uniform(20, 180),
            qed=rng.random(),
        )

    def modify_molecule(self, smiles: str, instruction: str) -> Dict[str, str]:
        instruction = (instruction or "").lower()
        new_smiles = smiles
        if "add methyl" in instruction or "methylate" in instruction:
            new_smiles = smiles + "C"
        elif "add fluorine" in instruction or "fluorinate" in instruction:
            new_smiles = smiles + "F"
        elif "add hydroxyl" in instruction or "hydroxylate" in instruction:
            new_smiles = smiles + "O"
        elif "add chlorine" in instruction or "chlorinate" in instruction:
            new_smiles = smiles + "Cl"
        elif "remove" in instruction:
            new_smiles = smiles[:-1] if len(smiles) > 6 else smiles
        else:
            new_smiles = smiles + "C"
        if RDKIT_AVAILABLE and Chem.MolFromSmiles(new_smiles) is None:
            new_smiles = smiles
        return {"smiles": new_smiles, "instruction": instruction}

    def run_docking(self, smiles: str, target: str) -> Dict[str, float]:
        rng = _rng_for(smiles, f"dock:{target}")
        affinity = self.predict_binding_affinity(smiles, target)
        base = -7.0 - 3.0 * affinity["score"] + rng.gauss(0, 0.8)
        clamped = max(-12.0, min(-4.0, base))
        quality = (-clamped - 4.0) / 8.0
        return {"docking_score": float(base), "quality": float(quality), "target": target}


_DEFAULT_LIBRARY: List[str] = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "C1=CC(=CC=C1CCN)O",
    "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "Cc1ccc2nc(-c3ccc(NS(C)(=O)=O)cc3)sc2c1",
    "Clc1ccc2[nH]c(=O)c(Cc3ccccc3)nc2c1",
    "OC(=O)Cc1ccc(O)cc1",
    "CN1CCC[C@H]1c1cccnc1",
    "CC(C(=O)O)N",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc2c(c1)nc1ccc(Cl)cc1c2N",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "OC(=O)c1ccccc1O",
    "CCCCC(=O)Nc1ccc(O)cc1",
    "CC(C)c1ccc(cc1)C(C)C(=O)O",
]
