from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataProvider(ABC):
    @abstractmethod
    def get_targets_for_disease(self, disease: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_compounds(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def search_literature(self, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError
