"""FastAPI entrypoint for the Drug Discovery Sim Environment."""

from __future__ import annotations

from fastapi import FastAPI

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.openenv_compat import HTTPEnvServer
from drug_discovery_env.server.environment import DrugDiscoveryEnv


def create_app() -> FastAPI:
    app = FastAPI(title="Drug Discovery Sim Env (OpenEnv)")

    server = HTTPEnvServer(
        env=DrugDiscoveryEnv,
        action_cls=DrugDiscoveryAction,
        observation_cls=DrugDiscoveryObservation,
        max_concurrent_envs=8,
    )
    server.register_routes(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
