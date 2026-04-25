from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.data_provider.factory import build_data_provider
from drug_discovery_env.data_provider.hybrid_provider import HybridDataProvider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider
from drug_discovery_env.data_provider.local_provider import LocalSnapshotProvider


def test_factory_modes() -> None:
    settings = get_settings().model_copy(deep=True)

    settings.data.mode = DataSourceMode.LOCAL_ONLY
    assert isinstance(build_data_provider(settings), LocalSnapshotProvider)

    settings.data.mode = DataSourceMode.HYBRID
    assert isinstance(build_data_provider(settings), HybridDataProvider)

    settings.data.mode = DataSourceMode.LIVE_ONLY
    assert isinstance(build_data_provider(settings), LiveAPIProvider)
