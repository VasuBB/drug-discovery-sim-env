from drug_discovery_env.training.grpo_trainer import run_grpo_if_available


def test_grpo_readiness_path() -> None:
    out = run_grpo_if_available(enable_actual_training=False, num_episodes=1, disease="Type 2 Diabetes")
    assert out["status"] in {"trl_ready", "dry_run_no_trl"}
    assert out["num_samples"] > 0
