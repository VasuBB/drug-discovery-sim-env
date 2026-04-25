"""Drug discovery simulation environment — OpenEnv hackathon submission."""

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.core.state import GameState

# Keep package import setup-safe: openenv may not be installed yet during bootstrap.
try:
    from drug_discovery_env.core.models import (
        DrugDiscoveryAction,
        DrugDiscoveryObservation,
        DrugDiscoveryState,
    )
except ModuleNotFoundError:  # pragma: no cover
    DrugDiscoveryAction = None  # type: ignore[assignment]
    DrugDiscoveryObservation = None  # type: ignore[assignment]
    DrugDiscoveryState = None  # type: ignore[assignment]

__all__ = [
    "DataSourceMode",
    "DrugDiscoveryAction",
    "DrugDiscoveryObservation",
    "DrugDiscoveryState",
    "GameState",
    "Settings",
    "get_settings",
]
