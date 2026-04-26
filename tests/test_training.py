from drug_discovery_env.config.settings import DataSourceMode, get_settings
from drug_discovery_env.training.grpo_trainer import (
    _make_replay_reward_func,
    run_training_dry_run,
)
from drug_discovery_env.training.rollout_generator import generate_rollouts


def test_training_dry_run() -> None:
    report = run_training_dry_run(num_episodes=1, disease="Type 2 Diabetes")
    assert report.num_samples > 0
    assert 0.0 <= report.reward_mean <= 1.0


def test_training_dataset_uses_full_prompt_and_history() -> None:
    samples = generate_rollouts(num_episodes=1, disease="Type 2 Diabetes")
    assert "Return one JSON object only." in samples[0].prompt
    assert isinstance(samples[0].history, list)


def test_replay_reward_func_varies_by_completion() -> None:
    sample = generate_rollouts(num_episodes=1, disease="Type 2 Diabetes")[0]

    settings = get_settings().model_copy(deep=True)
    settings.data.mode = DataSourceMode.LIVE_ONLY
    reward_func = _make_replay_reward_func(settings)

    good = reward_func(
        completions=[
            '{"tool":"select_target","params":{"disease":"Type 2 Diabetes"},"target_compound_id":null,"reasoning":"Pick the best target."}'
        ],
        disease=[sample.disease],
        history=[sample.history],
    )[0]
    bad = reward_func(
        completions=[
            '{"tool":"pause_and_review_all","params":{},"target_compound_id":null,"reasoning":"Wait."}'
        ],
        disease=[sample.disease],
        history=[sample.history],
    )[0]

    assert good != bad
