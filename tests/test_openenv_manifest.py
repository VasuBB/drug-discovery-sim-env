from pathlib import Path


def test_openenv_manifest_exists() -> None:
    manifest = Path('openenv.yaml')
    assert manifest.exists()
    text = manifest.read_text(encoding='utf-8')
    assert 'runtime: fastapi' in text
    assert 'app: drug_discovery_env.server.app:app' in text
