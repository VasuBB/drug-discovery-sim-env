from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from drug_discovery_env.data_provider.schemas import (
    CompoundSnapshotRow,
    LiteratureSnapshotRow,
    TargetSnapshotRow,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(data_dir: Path) -> list[dict[str, object]]:
    manifest_path = data_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("snapshot_manifest.json not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        file_path = data_dir / str(item["file"])
        if not file_path.exists():
            raise FileNotFoundError(f"Snapshot file missing: {file_path}")
        observed = file_sha256(file_path)
        expected = str(item["sha256"])
        if observed != expected:
            raise ValueError(f"Checksum mismatch for {file_path.name}")
    return manifest


def validate_snapshot_payloads(data_dir: Path) -> None:
    target_payload = json.loads((data_dir / "targets_snapshot.json").read_text(encoding="utf-8"))
    compound_payload = json.loads((data_dir / "compound_library.json").read_text(encoding="utf-8"))
    literature_payload = json.loads((data_dir / "literature_snapshot.json").read_text(encoding="utf-8"))

    TypeAdapter(list[TargetSnapshotRow]).validate_python(target_payload)
    TypeAdapter(list[CompoundSnapshotRow]).validate_python(compound_payload)
    TypeAdapter(list[LiteratureSnapshotRow]).validate_python(literature_payload)
