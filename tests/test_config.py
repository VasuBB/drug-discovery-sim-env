from drug_discovery_env.config.settings import DataSourceMode, get_settings


def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.app.max_steps == 50
    assert settings.data.mode in {DataSourceMode.CACHED_TARGETS_LIVE_TOOLS, DataSourceMode.LIVE_ONLY}
    assert "select_target" in settings.tools.costs
    assert settings.training.group_size >= 1
    assert settings.evaluation.morgan_n_bits > 0
