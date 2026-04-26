import json
from pathlib import Path
import shutil

from drug_discovery_env.server.environment import DrugDiscoveryEnv
from drug_discovery_env.training.model_policy import action_from_text, resolve_model_artifact_path
from drug_discovery_env.training.policy_evaluator import evaluate_in_process, scripted_action


def test_action_from_text_fallback() -> None:
    action = action_from_text("not json")
    assert action.tool == "pause_and_review_all"


def test_scripted_policy_evaluation_runs() -> None:
    env = DrugDiscoveryEnv()
    metrics = evaluate_in_process(
        env=env,
        disease="Type 2 Diabetes",
        episodes=1,
        choose_action=scripted_action,
    )
    assert len(metrics) == 1
    assert metrics[0].final_stage_index >= 0


def test_resolve_model_artifact_path_prefers_checkpoint() -> None:
    root = Path.cwd() / ".test-artifacts" / "outputs" / "grpo"
    checkpoint = root / "checkpoint-20"
    shutil.rmtree(root.parent.parent, ignore_errors=True)
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")

    resolved = resolve_model_artifact_path(str(root))
    assert resolved == checkpoint

    shutil.rmtree(root.parent.parent, ignore_errors=True)
