from drug_discovery_env.config.settings import get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction
from drug_discovery_env.core.state import CompoundRecord, GameState
from drug_discovery_env.rewards.aggregator import RewardEngine


def test_reward_engine_contract() -> None:
    settings = get_settings()
    state = GameState(disease="Type 2 Diabetes")
    state.compound_ledger["A"] = CompoundRecord(
        smiles="A",
        potency=0.8,
        selectivity=0.7,
        safety=0.75,
        synthesizability=0.6,
        novelty=0.5,
        developability=0.65,
        metadata={"admet": {"herg_prob": 0.2}},
    )
    action = DrugDiscoveryAction(tool="search_literature", params={}, reasoning="hypothesis with uncertainty tradeoff", evidence_ids=["pmid_1"])
    reward = RewardEngine(settings).compute(state, action)
    assert 0.0 <= reward.total <= 1.0
