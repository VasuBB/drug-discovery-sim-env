"""Compatibility layer for OpenEnv import paths.

Tries to import the real `openenv` / `openenv_core` packages; if neither is
installed, falls back to a self-contained, *fully working* shim that mounts
the standard OpenEnv HTTP routes (`/reset`, `/step`, `/state`) onto a
FastAPI app. The shim is what powers the public Hugging Face Space, where
`openenv` is not (currently) on PyPI.

Wire format (matches OpenEnv core conventions):
    POST /reset    body: {<reset kwargs>}                  -> StepResult JSON
    POST /step     body: {action: {...}, ...}              -> StepResult JSON
    GET  /state    -> State JSON
    POST /close    -> {"closed": true}

`StepResult` JSON layout:
    {"observation": {...}, "reward": float|null, "done": bool}
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Generic, Optional, Type, TypeVar

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
    # ---------------- shared base classes (Pydantic) ----------------
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
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

        def close(self) -> None:
            return None

    # ---------------- real route-mounting HTTPEnvServer ----------------
    class HTTPEnvServer:  # type: ignore[no-redef]
        """Mounts /reset /step /state /close on a FastAPI app.

        Concurrency model:
          - SUPPORTS_CONCURRENT_SESSIONS == False -> one shared env instance,
            protected by a lock (serializes /reset and /step).
          - SUPPORTS_CONCURRENT_SESSIONS == True  -> per-session env instances
            keyed by `session_id`, created on demand at /reset.
        """

        def __init__(
            self,
            env: Type[Any],
            action_cls: Type[BaseModel],
            observation_cls: Type[BaseModel],
            state_cls: Optional[Type[BaseModel]] = None,
            max_concurrent_envs: int = 8,
        ) -> None:
            self.env_cls = env
            self.action_cls = action_cls
            self.observation_cls = observation_cls
            self.state_cls = state_cls
            self.max_concurrent_envs = max_concurrent_envs

            self._concurrent = bool(getattr(env, "SUPPORTS_CONCURRENT_SESSIONS", False))
            self._lock = threading.Lock()
            self._sessions: Dict[str, Any] = {}
            self._default_env: Optional[Any] = None

        # ---- session helpers ----
        def _get_default_env(self) -> Any:
            if self._default_env is None:
                self._default_env = self.env_cls()
            return self._default_env

        def _get_session_env(self, session_id: str) -> Any:
            env = self._sessions.get(session_id)
            if env is None:
                if len(self._sessions) >= self.max_concurrent_envs:
                    # Evict the oldest session (FIFO) to bound memory.
                    oldest = next(iter(self._sessions))
                    self._sessions.pop(oldest, None)
                env = self.env_cls()
                self._sessions[session_id] = env
            return env

        def _resolve_env(self, session_id: Optional[str]) -> Any:
            if not self._concurrent:
                return self._get_default_env()
            sid = session_id or str(uuid.uuid4())
            return self._get_session_env(sid)

        @staticmethod
        def _to_jsonable(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump()
            return value

        def _step_result(self, observation: Any) -> Dict[str, Any]:
            obs_payload = self._to_jsonable(observation) or {}
            reward = None
            done = False
            if isinstance(obs_payload, dict):
                reward = obs_payload.get("reward")
                done = bool(obs_payload.get("done", False))
            return {"observation": obs_payload, "reward": reward, "done": done}

        # ---- FastAPI wiring ----
        def register_routes(self, app: Any) -> None:
            # Import at call time so the shim itself doesn't hard-require
            # FastAPI just to be importable (e.g. in tool-only CI runs).
            from fastapi import Body, HTTPException

            action_cls = self.action_cls

            @app.post("/reset")
            async def _reset(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
                if not isinstance(body, dict):
                    raise HTTPException(status_code=400, detail="reset body must be a JSON object")
                kwargs = dict(body)
                session_id = kwargs.pop("session_id", None)
                with self._lock:
                    env = self._resolve_env(session_id)
                    try:
                        observation = env.reset(**kwargs)
                    except TypeError as exc:
                        raise HTTPException(status_code=422, detail=f"reset kwargs invalid: {exc}")
                    except ValueError as exc:
                        raise HTTPException(status_code=422, detail=str(exc))
                return self._step_result(observation)

            @app.post("/step")
            async def _step(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
                if not isinstance(body, dict):
                    raise HTTPException(status_code=400, detail="step body must be a JSON object")
                payload = dict(body)
                session_id = payload.pop("session_id", None)
                action_payload = payload.get("action", payload)
                try:
                    action = action_cls.model_validate(action_payload)
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"invalid action: {exc}")
                with self._lock:
                    env = self._resolve_env(session_id)
                    try:
                        observation = env.step(action)
                    except RuntimeError as exc:
                        raise HTTPException(status_code=409, detail=str(exc))
                return self._step_result(observation)

            @app.get("/state")
            async def _state(session_id: Optional[str] = None) -> Dict[str, Any]:
                with self._lock:
                    env = self._resolve_env(session_id)
                    try:
                        state_obj = env.state
                    except RuntimeError as exc:
                        raise HTTPException(status_code=409, detail=str(exc))
                payload = self._to_jsonable(state_obj)
                return payload if isinstance(payload, dict) else {"state": payload}

            @app.post("/close")
            async def _close(session_id: Optional[str] = None) -> Dict[str, Any]:
                with self._lock:
                    if session_id and session_id in self._sessions:
                        env = self._sessions.pop(session_id)
                        try:
                            env.close()
                        except Exception:
                            pass
                        return {"closed": True, "session_id": session_id}
                    if self._default_env is not None:
                        try:
                            self._default_env.close()
                        except Exception:
                            pass
                        self._default_env = None
                        return {"closed": True}
                return {"closed": False}

            # Mark routes as registered so app.py can assert success at boot.
            setattr(app.state, "openenv_routes_registered", True)

    # ---------------- minimal HTTP client ----------------
    class _StepResult(Generic[O]):
        def __init__(self, observation: Any, reward: Any, done: bool) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    class EnvClient(Generic[A, O, S]):  # type: ignore[no-redef]
        def __init__(self, base_url: str = "") -> None:
            self.base_url = base_url.rstrip("/")
            self._session = None

        # --- subclass override hooks (kept compatible with client.py) ---
        def _step_payload(self, action: Any) -> Dict[str, Any]:
            if isinstance(action, BaseModel):
                return action.model_dump()
            if isinstance(action, dict):
                return action
            return {"action": action}

        def _parse_result(self, payload: Dict[str, Any]) -> Any:
            obs = payload.get("observation", payload) or {}
            return _StepResult(observation=obs, reward=payload.get("reward"),
                               done=bool(payload.get("done", False)))

        def _parse_state(self, payload: Dict[str, Any]) -> Any:
            return payload

        # --- transport ---
        def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            import requests
            url = f"{self.base_url}{path}"
            r = requests.request(method, url, json=json_body, timeout=60)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {}

        # --- API ---
        def reset(self, **kwargs: Any) -> Any:
            return self._parse_result(self._request("POST", "/reset", kwargs))

        def step(self, action: Any) -> Any:
            return self._parse_result(self._request("POST", "/step", self._step_payload(action)))

        def state(self) -> Any:
            return self._parse_state(self._request("GET", "/state"))

        def close(self) -> Dict[str, Any]:
            return self._request("POST", "/close")

        def sync(self) -> "EnvClient[A, O, S]":
            return self

        def __enter__(self) -> "EnvClient[A, O, S]":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            try:
                self.close()
            except Exception:
                pass
            return False


__all__ = [
    "Action",
    "EnvClient",
    "Environment",
    "HTTPEnvServer",
    "Observation",
    "State",
]
