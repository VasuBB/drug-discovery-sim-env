from drug_discovery_env.config.settings import get_settings
from drug_discovery_env.core.models import DrugDiscoveryAction
from drug_discovery_env.core.state import CompoundRecord, GameState
from drug_discovery_env.rewards.aggregator import RewardEngine
from drug_discovery_env.rewards.budget_efficiency import BudgetEfficiencyReward
from drug_discovery_env.rewards.oversight import OversightPenalty
from drug_discovery_env.rewards.reasoning import ReasoningReward
from drug_discovery_env.rewards.terminal import TerminalReward


def _record(**kwargs) -> CompoundRecord:
    base = dict(
        smiles="CCO",
        potency=0.7,
        selectivity=0.5,
        safety=0.7,
        synthesizability=0.6,
        novelty=0.5,
        developability=0.6,
        binding_affinity_nM=50.0,
        admet={"ro5_pass": True, "pains": False, "tox_flag": False, "tox_score": 0.1, "herg_prob": 0.1},
        metadata={"admet": {"herg_prob": 0.1}},
    )
    base.update(kwargs)
    return CompoundRecord(**base)


def test_reward_engine_contract_in_range() -> None:
    settings = get_settings()
    state = GameState(disease="Type 2 Diabetes")
    state.compound_ledger["A"] = _record()
    state.advanced_compound_id = state.compound_ledger["A"].id = "C001"
    state.compound_id_index["C001"] = "A"
    action = DrugDiscoveryAction(tool="search_literature", params={}, reasoning="hypothesis with uncertainty tradeoff", evidence_ids=["pmid_1"])
    reward = RewardEngine(settings).compute(state, action, terminal=True)
    assert 0.0 <= reward.total <= 1.0
    assert reward.terminal_compound > 0.0


def test_terminal_safety_floor_zeroes_score() -> None:
    settings = get_settings()
    bad = _record(metadata={"admet": {"herg_prob": 0.9}}, admet={"ro5_pass": True, "pains": False, "herg_prob": 0.9})
    assert TerminalReward(settings).score(bad) == 0.0
    pains = _record(admet={"ro5_pass": True, "pains": True, "herg_prob": 0.0}, metadata={"admet": {"herg_prob": 0.0}})
    assert TerminalReward(settings).score(pains) == 0.0


def test_terminal_monotonic_in_potency() -> None:
    settings = get_settings()
    weak = TerminalReward(settings).score(_record(binding_affinity_nM=10000.0))
    strong = TerminalReward(settings).score(_record(binding_affinity_nM=1.0))
    assert strong >= weak


def test_budget_concave() -> None:
    r = BudgetEfficiencyReward()
    assert r.score(0.0, 100.0) == 0.0
    assert r.score(50.0, 100.0) > 0.5
    assert r.score(100.0, 100.0) == 1.0


def test_oversight_saturating() -> None:
    r = OversightPenalty()
    a = r.score(0, 1)
    b = r.score(0, 10)
    assert a < b < 1.0
    assert r.score(0, 0) == 0.0


def test_reasoning_smoothness() -> None:
    r = ReasoningReward()
    short = r.score(["binding selectivity"])
    longer = r.score(["binding selectivity uncertainty hypothesis novelty docking " * 4])
    assert longer >= short
