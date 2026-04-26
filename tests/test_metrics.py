from pathlib import Path

from drug_discovery_env.training.metrics import (
    DiseaseMetricRecord,
    aggregate_report,
    best_tanimoto,
    mean_tanimoto,
    tanimoto,
    write_per_disease,
)


def test_tanimoto_self_is_one() -> None:
    assert tanimoto("CCO", "CCO") >= 0.99


def test_best_and_mean_tanimoto() -> None:
    refs = ["CCO", "CCN", "COCC"]
    b = best_tanimoto("CCO", refs, radius=2, n_bits=512)
    m = mean_tanimoto("CCO", refs, radius=2, n_bits=512)
    assert 0.0 <= m <= b <= 1.0


def test_aggregate_report(tmp_path: Path) -> None:
    records = [
        DiseaseMetricRecord(
            disease=f"D{i}",
            target="T",
            nominated_smiles="CCO",
            terminal_reward=0.5,
            total_reward=0.6,
            reward_breakdown={},
            final_stage_idx=4,
            stage_completed=True,
            budget_remaining_frac=0.4,
            admet_pass=True,
            oversight_violations=0,
            mean_reasoning_depth=0.3,
            tanimoto_to_known_max=0.6,
            tanimoto_to_known_mean=0.3,
            precision_at_1=1.0,
            n_known_drugs=2,
            steps=10,
            terminated_reason="ok",
        )
        for i in range(3)
    ]
    per_disease = write_per_disease(records, tmp_path / "per.jsonl")
    report = aggregate_report(records, per_disease)
    assert report["n_test_diseases"] == 3
    assert report["precision_at_1"] == 1.0
    assert report["admet_pass_rate"] == 1.0
    assert report["mean_terminal_reward"] == 0.5
