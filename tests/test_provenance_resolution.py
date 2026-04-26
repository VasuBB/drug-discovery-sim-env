from drug_discovery_env.server.environment import DrugDiscoveryEnv


def test_search_compounds_provenance_not_simulation() -> None:
    env = DrugDiscoveryEnv()
    _ = env.reset("Type 2 Diabetes")
    _ = env.step("<reasoning>select target</reasoning><tool>select_target</tool><params>{}</params>")
    obs = env.step("<reasoning>search hits</reasoning><tool>search_compounds</tool><params>{\"min_qed\":0.4}</params>")
    assert obs.provenance is not None
    assert obs.provenance.source == "live"


def test_search_literature_provenance_from_docs() -> None:
    env = DrugDiscoveryEnv()
    _ = env.reset("Type 2 Diabetes")
    obs = env.step("<reasoning>find evidence</reasoning><tool>search_literature</tool><params>{\"query\":\"INSR diabetes safety\"}</params>")
    assert obs.provenance is not None
    assert obs.provenance.source == "live"
