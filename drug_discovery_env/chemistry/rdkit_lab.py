"""RDKit-backed cheminformatics simulator.

Ported from the lightweight `lab_simulator` of the `main` branch and merged
with this branch's structured CompoundRecord / Settings architecture.

Provides plausible (not real) chemistry feedback for:
  - binding affinity (nM + score)
  - ADMET (Lipinski RO5, PAINS substructures, tox SMARTS, hERG proxy, QED)
  - molecular modification
  - docking score

Gracefully degrades when RDKit is not installed: deterministic SMILES-hash
based pseudo-random scores so unit tests and CI still run.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on RDKit-less CI
    RDKIT_AVAILABLE = False


# Substructure libraries used by the toxicology / PAINS heuristics.
PAINS_SMARTS = (
    "[OX1]=[CX3][CX3]=[CX3][OX1]",
    "c1ccc2c(c1)C(=O)c1ccccc1C2=O",
    "[CX3](=O)[CX3](=O)",
)
TOX_SMARTS = (
    "[Cl,Br,I][CX4]",
    "C(=O)Cl",
    "[NX3](=O)=O",
    "S(=O)(=O)Cl",
)
# Naive hERG-liability flags: large lipophilic bases / extended aromatic systems.
HERG_SMARTS = (
    "c1ccc(cc1)C(C)(C)N",
    "c1ccc2c(c1)cccc2",
)


def _smiles_seed(smiles: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{salt}|{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng_for(smiles: str, salt: str = "") -> random.Random:
    return random.Random(_smiles_seed(smiles, salt))


@dataclass
class ADMETResult:
    """Plausible drug-likeness profile for a single molecule."""

    ro5_pass: bool
    pains: bool
    tox_flag: bool
    tox_score: float
    herg_prob: float
    logp: float
    mw: float
    hbd: int
    hba: int
    tpsa: float
    qed: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ro5_pass": bool(self.ro5_pass),
            "pains": bool(self.pains),
            "tox_flag": bool(self.tox_flag),
            "tox_score": float(self.tox_score),
            "herg_prob": float(self.herg_prob),
            "logp": float(self.logp),
            "mw": float(self.mw),
            "hbd": int(self.hbd),
            "hba": int(self.hba),
            "tpsa": float(self.tpsa),
            "qed": float(self.qed),
        }


class RDKitLab:
    """Stateless cheminformatics simulator. Safe to share across episodes."""

    def predict_binding_affinity(self, smiles: str, target: str) -> dict[str, float]:
        rng = _rng_for(smiles, f"affinity:{target}")
        base = rng.gauss(2.0, 1.0)
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                base += 0.5 * abs(mw - 350) / 350
                base += 0.4 * abs(logp - 2.5) / 2.5
        affinity_nM = max(0.1, 10**base)
        # Higher score = better; affinities below ~10nM saturate near 1.
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
            pains = self._has_any_match(mol, PAINS_SMARTS)
            tox_hits = sum(1 for s in TOX_SMARTS if self._has_match(mol, s))
            tox_score = min(1.0, tox_hits / 3.0)
            tox_flag = tox_score >= 0.34
            herg_hits = sum(1 for s in HERG_SMARTS if self._has_match(mol, s))
            # Simple proxy: lipophilic + aromatic flags raise hERG probability.
            herg_prob = max(
                0.0,
                min(1.0, 0.15 * herg_hits + 0.05 * max(0.0, logp - 3.0)),
            )
            return ADMETResult(
                ro5_pass=ro5_pass,
                pains=pains,
                tox_flag=tox_flag,
                tox_score=tox_score,
                herg_prob=float(herg_prob),
                logp=float(logp),
                mw=float(mw),
                hbd=int(hbd),
                hba=int(hba),
                tpsa=float(tpsa),
                qed=float(qed),
            )
        return self._fallback_admet(smiles)

    def modify_molecule(self, smiles: str, instruction: str) -> dict[str, str]:
        instruction = (instruction or "").lower()
        new_smiles = smiles
        if "methyl" in instruction:
            new_smiles = smiles + "C"
        elif "fluor" in instruction:
            new_smiles = smiles + "F"
        elif "hydroxyl" in instruction or "oh" in instruction:
            new_smiles = smiles + "O"
        elif "chlor" in instruction:
            new_smiles = smiles + "Cl"
        elif "remove" in instruction and len(smiles) > 6:
            new_smiles = smiles[:-1]
        else:
            new_smiles = smiles + "C"
        if RDKIT_AVAILABLE and Chem.MolFromSmiles(new_smiles) is None:
            new_smiles = smiles
        return {"smiles": new_smiles, "instruction": instruction}

    def run_docking(self, smiles: str, target: str) -> dict[str, float]:
        rng = _rng_for(smiles, f"dock:{target}")
        affinity = self.predict_binding_affinity(smiles, target)
        base = -7.0 - 3.0 * affinity["score"] + rng.gauss(0, 0.8)
        clamped = max(-12.0, min(-4.0, base))
        quality = (-clamped - 4.0) / 8.0
        return {"docking_score": float(base), "quality": float(quality), "target": target}

    # ------------------------------------------------------------------

    def _has_match(self, mol: Any, smarts: str) -> bool:
        pat = Chem.MolFromSmarts(smarts) if RDKIT_AVAILABLE else None
        return bool(pat is not None and mol.HasSubstructMatch(pat))

    def _has_any_match(self, mol: Any, smarts_list) -> bool:
        return any(self._has_match(mol, s) for s in smarts_list)

    def _fallback_admet(self, smiles: str, invalid: bool = False) -> ADMETResult:
        rng = _rng_for(smiles, "admet")
        if invalid:
            return ADMETResult(False, True, True, 1.0, 0.9, 6.0, 600, 8, 12, 200, 0.05)
        return ADMETResult(
            ro5_pass=rng.random() > 0.4,
            pains=rng.random() < 0.15,
            tox_flag=rng.random() < 0.25,
            tox_score=rng.random(),
            herg_prob=rng.random() * 0.6,
            logp=rng.uniform(-1, 6),
            mw=rng.uniform(180, 600),
            hbd=rng.randint(0, 7),
            hba=rng.randint(1, 12),
            tpsa=rng.uniform(20, 180),
            qed=rng.random(),
        )
