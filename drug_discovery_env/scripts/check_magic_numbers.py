from __future__ import annotations

import re
from pathlib import Path


NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
ALLOWLIST = {"0", "1", "2", "3", "4", "5", "10", "50", "100"}


def scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "settings." in line or line.strip().startswith("#"):
            continue
        for token in NUMERIC_PATTERN.findall(line):
            if token not in ALLOWLIST and "test" not in str(path):
                violations.append(f"{path}:{idx}:{token}")
    return violations


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [p for p in root.rglob("*.py") if ".venv" not in str(p)]
    violations = []
    for path in files:
        violations.extend(scan_file(path))

    if violations:
        print("Magic-number check warnings:")
        for v in violations[:80]:
            print(v)
        raise SystemExit(1)
    print("Magic-number check passed")


if __name__ == "__main__":
    main()
