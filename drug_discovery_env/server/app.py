"""FastAPI entrypoint for the Drug Discovery Sim Environment."""

from __future__ import annotations

from fastapi import FastAPI

from drug_discovery_env.core.models import (
    DrugDiscoveryAction,
    DrugDiscoveryObservation,
    DrugDiscoveryState,
)
from drug_discovery_env.openenv_compat import HTTPEnvServer
from drug_discovery_env.server.environment import DrugDiscoveryEnv


REQUIRED_ROUTES = {"/reset", "/step", "/state"}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Drug Discovery Sim Env (OpenEnv)",
        version="0.1.0",
        description=(
            "OpenEnv-compatible HTTP environment for a 50-step simulated drug "
            "discovery campaign. Standard routes: POST /reset, POST /step, "
            "GET /state, POST /close. See /docs for the live OpenAPI schema."
        ),
    )

    server = HTTPEnvServer(
        env=DrugDiscoveryEnv,
        action_cls=DrugDiscoveryAction,
        observation_cls=DrugDiscoveryObservation,
        state_cls=DrugDiscoveryState,
        max_concurrent_envs=8,
    )
    server.register_routes(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # Fail loud: if HTTPEnvServer did not actually mount /reset /step /state
    # (e.g. a no-op shim slipped in), we want the container to crash at boot
    # rather than silently ship a server that 404s every env call.
    mounted = {getattr(r, "path", None) for r in app.routes}
    missing = REQUIRED_ROUTES - mounted
    if missing:
        raise RuntimeError(
            f"OpenEnv routes not mounted: {sorted(missing)}. "
            "HTTPEnvServer.register_routes() did not register the standard "
            "/reset /step /state surface — refusing to start."
        )

    return app


app = create_app()
