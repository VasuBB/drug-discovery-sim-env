from drug_discovery_env.server.environment import DrugDiscoveryEnv


def test_tool_contracts_have_provenance_and_result() -> None:
    env = DrugDiscoveryEnv()
    _ = env.reset("Type 2 Diabetes")
    obs = env.step(
        "<reasoning>evidence-driven target pick</reasoning><tool>select_target</tool><params>{\"disease\":\"Type 2 Diabetes\"}</params>"
    )
    assert obs.tool_result is not None
    assert obs.provenance is not None
