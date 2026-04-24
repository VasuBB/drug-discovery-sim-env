from __future__ import annotations

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


class ToxicologistAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: GameState) -> list[str]:
        alerts: list[str] = []
        threshold = self.settings.agents.toxicologist_alert_threshold
        for smiles, rec in list(state.compound_ledger.items())[-5:]:
            herg = float(rec.metadata.get("admet", {}).get("herg_prob", 0.0))
            topo_risk = max(state.off_target_risk_profile.values()) if state.off_target_risk_profile else 0.0
            if herg > threshold or topo_risk > threshold:
                alerts.append(f"High cardiac/pathway risk for {smiles[:16]} (hERG={herg:.2f}, topo={topo_risk:.2f}).")
        return alerts
