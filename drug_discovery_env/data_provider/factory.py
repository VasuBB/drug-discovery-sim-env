from __future__ import annotations

from drug_discovery_env.config.settings import DataSourceMode, Settings
from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.cached_provider import CachedTargetsLiveToolsProvider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider


def build_data_provider(settings: Settings) -> DataProvider:
    mode = settings.data.mode
    if mode == DataSourceMode.CACHED_TARGETS_LIVE_TOOLS:
        return CachedTargetsLiveToolsProvider(settings)
    if mode == DataSourceMode.LIVE_ONLY:
        return LiveAPIProvider(settings)
    raise ValueError(f"Unsupported data mode: {mode!r}")
