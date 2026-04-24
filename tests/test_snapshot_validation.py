from pathlib import Path

from drug_discovery_env.data_provider.snapshot_utils import validate_manifest, validate_snapshot_payloads


def test_snapshot_manifest_and_schema() -> None:
    root = Path("drug_discovery_env/data")
    validate_snapshot_payloads(root)
    manifest = validate_manifest(root)
    assert len(manifest) >= 3
