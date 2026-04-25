from drug_discovery_env.training.grpo_trainer import run_training_dry_run


def test_training_dry_run() -> None:
    report = run_training_dry_run(num_episodes=1)
    assert report.num_samples > 0
    assert 0.0 <= report.reward_mean <= 1.0
