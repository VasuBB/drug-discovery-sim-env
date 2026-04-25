"""Toxicologist sub-agent — PAINS / RO5 / tox-flag block warnings."""

from __future__ import annotations

from typing import Dict, List

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


def _msg(severity: str, message: str) -> Dict[str, str]:
    return {"agent": "toxicologist", "severity": severity, "message": message}


class ToxicologistAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: GameState) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        threshold = self.settings.agents.toxicologist_alert_threshold

        for rec in state.compound_ledger.values():
            admet = rec.admet or {}
            if not admet:
                continue
            cid = rec.id or rec.smiles[:12]
            if admet.get("pains"):
                out.append(_msg("block", f"Compound {cid} hits a PAINS substructure — do not advance."))
            if admet.get("tox_flag"):
                out.append(_msg(
                    "warn",
                    f"Compound {cid} carries a toxic substructure (score {admet.get('tox_score', 0):.2f}).",
                ))
            if admet.get("ro5_pass") is False:
                out.append(_msg("warn", f"Compound {cid} fails Lipinski Rule-of-Five."))

            herg = float(rec.metadata.get("admet", {}).get("herg_prob", 0.0))
            topo_risk = max(state.off_target_risk_profile.values()) if state.off_target_risk_profile else 0.0
            if herg > threshold or topo_risk > threshold:
                out.append(_msg(
                    "warn",
                    f"High cardiac/pathway risk for {cid} (hERG={herg:.2f}, topo={topo_risk:.2f}).",
                ))

        if state.advanced_compound_id:
            adv = state.get_by_id(state.advanced_compound_id)
            if adv is not None:
                admet = adv.admet or {}
                if not admet:
                    out.append(_msg("warn", "Lead nominated without ADMET evaluation."))
                elif admet.get("pains") or admet.get("tox_flag"):
                    out.append(_msg(
                        "block",
                        f"Nominated lead {adv.id} has unresolved tox liability.",
                    ))
        return out
