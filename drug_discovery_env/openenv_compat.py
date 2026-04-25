from __future__ import annotations

"""Compatibility layer for OpenEnv import paths.

Supports both newer `openenv.core` and legacy `openenv_core` packages.
"""

try:  # Preferred import path (newer releases)
    from openenv.core import Action, EnvClient, Environment, HTTPEnvServer, Observation, State
except Exception:  # pragma: no cover - fallback path for legacy installs
    from openenv_core import Action, EnvClient, Environment, HTTPEnvServer, Observation, State

__all__ = [
    "Action",
    "EnvClient",
    "Environment",
    "HTTPEnvServer",
    "Observation",
    "State",
]
