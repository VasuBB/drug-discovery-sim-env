from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from drug_discovery_env.data_provider.snapshot_utils import validate_snapshot_payloads

SNAPSHOT_FILES = [
    "targets_snapshot.json",
    "compound_library.json",
    "literature_snapshot.json",
]


def sha256(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    validate_snapshot_payloads(root)
    manifest = []
    for name in SNAPSHOT_FILES:
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Snapshot {name} must be a list")
        manifest.append(
            {
                "file": name,
                "rows": len(payload),
                "sha256": sha256(path),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0.0",
            }
        )

    manifest_path = root / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
