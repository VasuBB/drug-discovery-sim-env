from drug_discovery_env.server.environment import DrugDiscoveryEnv


def test_literature_claim_grounding_contract() -> None:
    env = DrugDiscoveryEnv()
    _ = env.reset("Type 2 Diabetes")
    action = (
        "<reasoning>Ground claims from papers.</reasoning>"
        "<tool>search_literature</tool>"
        "<params>{\"query\":\"INSR selectivity safety\",\"claims\":[\"INSR modulation improves glucose control\",\"off-target selectivity reduces toxicity\"]}</params>"
    )
    obs = env.step(action)
    assert obs.tool_result is not None
    grounding = obs.tool_result.get("claim_grounding", [])
    assert len(grounding) == 2
    assert "snippet_id" in grounding[0]
