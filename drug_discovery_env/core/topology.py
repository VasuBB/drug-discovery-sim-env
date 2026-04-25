"""Pathway-graph topology engine for partial-observability world modeling."""

from __future__ import annotations

from collections import deque
from typing import Dict, Set

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.core.state import GameState


class TopologyEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def neighborhood(self, state: GameState, source_node: str) -> Set[str]:
        hops = int(self.settings.transitions.topology["neighborhood_hops"])
        visited: Set[str] = {source_node}
        queue: deque[tuple[str, int]] = deque([(source_node, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in state.pathway_graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    def propagate_risk(self, state: GameState, primary_target: str, base_risk: float) -> Dict[str, float]:
        risk_factor = float(self.settings.transitions.topology["risk_propagation_factor"])
        compensation = float(self.settings.transitions.topology["compensation_factor"])
        neighborhood = self.neighborhood(state, primary_target)
        out: Dict[str, float] = {}
        for idx, node in enumerate(sorted(neighborhood)):
            decay = max(0.0, 1.0 - idx * compensation * 0.1)
            out[node] = max(0.0, min(1.0, base_risk * risk_factor * decay))
        state.off_target_risk_profile.update(out)
        return out
