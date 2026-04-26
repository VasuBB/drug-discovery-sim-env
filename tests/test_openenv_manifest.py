from pathlib import Path


def test_openenv_manifest_exists() -> None:
    manifest = Path('openenv.yaml')
    assert manifest.exists()
    text = manifest.read_text(encoding='utf-8')
    assert 'runtime: fastapi' in text
    assert 'app: drug_discovery_env.server.app:app' in text
    # Port must match the Dockerfile (HF Spaces require 7860).
    assert 'port: 7860' in text
    # Manifest should advertise the standard OpenEnv route surface.
    for route in ('/reset', '/step', '/state', '/close'):
        assert route in text, f"missing {route} in openenv.yaml"
    # And the MCP tool surface.
    for tool in ('select_target', 'predict_affinity', 'evaluate_admet', 'advance_stage'):
        assert tool in text, f"missing tool {tool} in openenv.yaml"
