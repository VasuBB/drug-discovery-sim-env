"""FastAPI entrypoint for the Drug Discovery Sim Environment.

Exposes the standard OpenEnv routes (/ws, /reset, /step, /state, /health,
/web, /docs) by handing the environment class to `create_fastapi_app`.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
for p in (_PKG_DIR, _REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from openenv.core.env_server import create_fastapi_app  # noqa: E402

from drug_discovery_env import DrugDiscoveryEnvironment  # noqa: E402
from models import DrugDiscoveryAction, DrugDiscoveryObservation  # noqa: E402


app = create_fastapi_app(DrugDiscoveryEnvironment, DrugDiscoveryAction, DrugDiscoveryObservation)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run(
        "server.app:app" if __name__ != "__main__" else app,
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
