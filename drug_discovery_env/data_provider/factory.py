from __future__ import annotations

from drug_discovery_env.config.settings import DataSourceMode, Settings
from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.hybrid_provider import HybridDataProvider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider
from drug_discovery_env.data_provider.local_provider import LocalSnapshotProvider


def build_data_provider(settings: Settings) -> DataProvider:
    mode = settings.data.mode
    if mode == DataSourceMode.LIVE_ONLY:
        return LiveAPIProvider(settings)
    if mode == DataSourceMode.LOCAL_ONLY:
        return LocalSnapshotProvider(settings.data.local_data_dir)
    return HybridDataProvider(settings)
