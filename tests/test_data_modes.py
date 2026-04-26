import pytest

from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.data_provider.factory import build_data_provider
from drug_discovery_env.data_provider.live_provider import LiveAPIProvider


def test_factory_live_only() -> None:
    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.LIVE_ONLY
    assert isinstance(build_data_provider(settings), LiveAPIProvider)


def test_factory_rejects_non_live_mode() -> None:
    settings = get_settings().model_copy(deep=True)
    settings.data.mode = "hybrid"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="live_only"):
        build_data_provider(settings)
