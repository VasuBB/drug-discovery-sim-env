"""Compatibility layer for OpenEnv import paths.

Supports both newer `openenv.core` and legacy `openenv_core` packages, and
falls back to lightweight local shims when neither is installed (e.g. CI runs
that exercise the env logic without the full OpenEnv stack).
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

_loaded = False
try:
    from openenv.core import (  # type: ignore
        Action,
        EnvClient,
        Environment,
        HTTPEnvServer,
        Observation,
        State,
    )

    _loaded = True
except Exception:  # pragma: no cover
    try:
        from openenv_core import (  # type: ignore
            Action,
            EnvClient,
            Environment,
            HTTPEnvServer,
            Observation,
            State,
        )

        _loaded = True
    except Exception:
        _loaded = False


if not _loaded:
    # ---- minimal Pydantic-based shim ----
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: str | None = None
        step_count: int = 0

    class Environment(Generic[A, O, S]):  # type: ignore[no-redef]
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

        def __init__(self) -> None:
            pass

        def reset(self, *args: Any, **kwargs: Any) -> O:  # noqa: D401
            raise NotImplementedError

        def step(self, action: A, *args: Any, **kwargs: Any) -> O:
            raise NotImplementedError

        @property
        def state(self) -> S:
            raise NotImplementedError

    class HTTPEnvServer:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def register_routes(self, app: Any) -> None:
            return None

    class EnvClient(Generic[A, O, S]):  # type: ignore[no-redef]
        def __init__(self, base_url: str = "") -> None:
            self.base_url = base_url

        def sync(self) -> "EnvClient[A, O, S]":
            return self


__all__ = [
    "Action",
    "EnvClient",
    "Environment",
    "HTTPEnvServer",
    "Observation",
    "State",
]
