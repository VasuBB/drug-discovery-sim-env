"""Abstract base for all environment tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from drug_discovery_env.core.state import GameState


class Tool(ABC):
    name: str
    # Information-gain hint used by BudgetManager. Override per-tool.
    default_information_gain: float = 0.4

    @abstractmethod
    def execute(self, state: GameState, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
