from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from drug_discovery_env.core.state import GameState


class Tool(ABC):
    name: str

    @abstractmethod
    def execute(self, state: GameState, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
