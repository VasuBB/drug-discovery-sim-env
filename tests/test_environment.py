from drug_discovery_env.server.environment import DrugDiscoveryEnv


def test_environment_step_cycle() -> None:
    env = DrugDiscoveryEnv()
    _ = env.reset("Type 2 Diabetes")
    obs = env.step(
        "<reasoning>start target selection</reasoning><tool>select_target</tool><params>{\"disease\":\"Type 2 Diabetes\"}</params>"
    )
    assert obs.info["step"] == 1
    assert obs.reward_breakdown is not None
