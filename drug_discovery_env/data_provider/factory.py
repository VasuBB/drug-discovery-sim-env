from __future__ import annotations

from drug_discovery_env.config.settings import DataSourceMode, Settings
from drug_discovery_env.data_provider.base import DataProvider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider


def build_data_provider(settings: Settings) -> DataProvider:
    if settings.data.mode != DataSourceMode.LIVE_ONLY:
        raise ValueError("Only live_only data mode is supported")
    return LiveAPIProvider(settings)
