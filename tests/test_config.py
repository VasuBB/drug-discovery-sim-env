from drug_discovery_env.config.settings import DataSourceMode, get_settings


def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.app.max_steps == 50
    assert settings.data.mode in {DataSourceMode.HYBRID, DataSourceMode.LOCAL_ONLY, DataSourceMode.LIVE_ONLY}
    assert "select_target" in settings.tools.costs
