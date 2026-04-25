"""Drug discovery simulation environment package."""

from drug_discovery_env.config.settings import DataSourceMode, Settings, get_settings
from drug_discovery_env.core.state import GameState

# Keep package import setup-safe: openenv may not be installed yet when running
# bootstrap scripts such as setup_training_env.
try:
    from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
except ModuleNotFoundError:  # pragma: no cover - exercised in fresh bootstrap envs
    DrugDiscoveryAction = None  # type: ignore[assignment]
    DrugDiscoveryObservation = None  # type: ignore[assignment]

__all__ = [
    "DataSourceMode",
    "DrugDiscoveryAction",
    "DrugDiscoveryObservation",
    "GameState",
    "Settings",
    "get_settings",
]
