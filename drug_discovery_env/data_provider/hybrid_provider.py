from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from drug_discovery_env.config.settings import Settings
from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider
from drug_discovery_env.data_provider.local_provider import LocalSnapshotProvider


class HybridDataProvider(DataProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.live = LiveAPIProvider(settings)
        self.local = LocalSnapshotProvider(settings.data.local_data_dir)
        self._failures = 0
        self._last_resolution = "live"
        self._last_error: str | None = None

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _with_fallback(self, fn_live: Callable[[], Any], fn_local: Callable[[], Any]) -> Any:
        if self._failures >= self.settings.data.circuit_breaker_failures:
            self._last_resolution = "local"
            return fn_local()
        for _ in range(self.settings.data.retries + 1):
            try:
                result = fn_live()
                self._failures = 0
                self._last_resolution = "live"
                self._last_error = None
                return result
            except Exception as exc:
                self._failures += 1
                self._last_error = str(exc)
        self._last_resolution = "local"
        return fn_local()

    def get_targets_for_disease(self, disease: str) -> dict[str, Any]:
        result = self._with_fallback(
            lambda: self.live.get_targets_for_disease(disease),
            lambda: self.local.get_targets_for_disease(disease),
        )
        result.setdefault("timestamp", self._timestamp())
        result.setdefault("confidence", 0.5)
        result.setdefault("resolution", self._last_resolution)
        if self._last_error:
            result.setdefault("fallback_reason", self._last_error)
        return result

    def get_compounds(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self._with_fallback(
            lambda: self.live.get_compounds(query),
            lambda: self.local.get_compounds(query),
        )
        for row in rows:
            row.setdefault("resolution", self._last_resolution)
            if self._last_error:
                row.setdefault("fallback_reason", self._last_error)
        return rows

    def search_literature(self, query: str) -> list[dict[str, Any]]:
        docs = self._with_fallback(
            lambda: self.live.search_literature(query),
            lambda: self.local.search_literature(query),
        )
        for doc in docs:
            doc.setdefault("resolution", self._last_resolution)
            if self._last_error:
                doc.setdefault("fallback_reason", self._last_error)
        return docs
