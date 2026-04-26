"""Runtime config helper.

Resolves project-relative paths from the loaded :class:`Settings` and lets a
caller layer optional override yaml on top of :file:`config/defaults.yaml`.
This is a thin wrapper; the canonical schema still lives in
:mod:`drug_discovery_env.config.settings`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from drug_discovery_env.config.settings import Settings, get_settings, project_root


def resolve_path(relative_or_abs: str | Path) -> Path:
    p = Path(relative_or_abs)
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_settings(override_path: str | Path | None = None) -> Settings:
    """Load defaults.yaml, optionally merged with an override yaml."""
    if override_path is None:
        return get_settings()
    base_yaml = (project_root() / "config" / "defaults.yaml").read_text(encoding="utf-8")
    base = yaml.safe_load(base_yaml) or {}
    override = yaml.safe_load(Path(override_path).read_text(encoding="utf-8")) or {}
    merged = _deep_merge(base, override)
    return Settings(**merged)
