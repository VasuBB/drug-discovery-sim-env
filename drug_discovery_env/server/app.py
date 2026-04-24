from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from drug_discovery_env.core.models import DrugDiscoveryAction, DrugDiscoveryObservation
from drug_discovery_env.server.environment import DrugDiscoveryEnv


class ResetRequest(BaseModel):
    disease: str = "Type 2 Diabetes"


class StepRequest(BaseModel):
    action: str | None = None
    structured_action: DrugDiscoveryAction | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Drug Discovery Sim Env")
    env = DrugDiscoveryEnv()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset", response_model=DrugDiscoveryObservation)
    def reset(req: ResetRequest) -> DrugDiscoveryObservation:
        return env.reset(disease=req.disease)

    @app.post("/step", response_model=DrugDiscoveryObservation)
    def step(req: StepRequest) -> DrugDiscoveryObservation:
        if req.structured_action is not None:
            return env.step(req.structured_action)
        if req.action is None:
            raise ValueError("Either action or structured_action must be provided")
        return env.step(req.action)

    return app


app = create_app()
