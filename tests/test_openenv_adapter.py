from drug_discovery_env.server.openenv_adapter import create_openenv_adapter


def test_adapter_step_result_contract() -> None:
    adapter = create_openenv_adapter()
    _ = adapter.reset()
    out = adapter.step("<reasoning>select target</reasoning><tool>select_target</tool><params>{}</params>")
    assert out.observation is not None
    assert isinstance(out.reward, float)
    assert isinstance(out.done, bool)
    assert isinstance(out.info, dict)
