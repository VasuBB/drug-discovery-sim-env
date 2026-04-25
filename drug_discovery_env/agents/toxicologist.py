from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


class ToxicologistAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: GameState) -> list[str]:
        return [m["message"] for m in self.messages(state)]

    def messages(self, state: GameState) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        threshold = self.settings.agents.toxicologist_alert_threshold
        herg_floor = float(self.settings.reward.floors.get("herg_prob_max", 0.5))
        for smiles, rec in list(state.compound_ledger.items())[-5:]:
            admet = rec.metadata.get("admet", {}) or {}
            herg = float(admet.get("herg_prob", 0.0))
            topo_risk = max(state.off_target_risk_profile.values()) if state.off_target_risk_profile else 0.0
            if admet.get("pains"):
                out.append({
                    "agent": "toxicologist",
                    "severity": "block",
                    "message": f"Compound {smiles[:16]} matches a PAINS substructure -- do not advance.",
                })
            if herg > herg_floor:
                out.append({
                    "agent": "toxicologist",
                    "severity": "block",
                    "message": f"Compound {smiles[:16]} hERG probability {herg:.2f} exceeds floor {herg_floor:.2f}.",
                })
            elif herg > threshold or topo_risk > threshold:
                out.append({
                    "agent": "toxicologist",
                    "severity": "warn",
                    "message": (
                        f"High cardiac/pathway risk for {smiles[:16]} "
                        f"(hERG={herg:.2f}, topo={topo_risk:.2f})."
                    ),
                })
            if admet.get("ro5_pass") is False:
                out.append({
                    "agent": "toxicologist",
                    "severity": "warn",
                    "message": f"Compound {smiles[:16]} fails Lipinski Rule-of-Five.",
                })
        return out
