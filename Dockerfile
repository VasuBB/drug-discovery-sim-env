# Hugging Face Space (Docker SDK) — runs the Drug Discovery env server.
#
# We deliberately do NOT install torch / TRL / Unsloth here: the Space only
# serves the env over HTTP for judges, training stays on Kaggle/Colab.
# That keeps the image small enough for HF free-tier (CPU basic).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf-cache

# RDKit wheels need libxrender / libxext at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxrender1 \
        libxext6 \
        libsm6 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the env server needs.
COPY pyproject.toml README.md /app/
COPY drug_discovery_env /app/drug_discovery_env
COPY config /app/config
COPY data /app/data

# Slim runtime deps: no torch / trl / unsloth.
RUN pip install --upgrade pip \
 && pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.2" \
        "PyYAML>=6.0" \
        "numpy>=1.26" \
        "networkx>=3.2" \
        "requests>=2.32" \
        "rank-bm25>=0.2" \
        "rdkit>=2024.3.1" \
 && pip install --no-deps -e . \
 && pip install openenv || echo "[warn] openenv not on PyPI in this image; using bundled openenv_compat shim"

EXPOSE 7860

# Hugging Face Spaces require the app to listen on $PORT (default 7860).
CMD ["sh", "-c", "uvicorn drug_discovery_env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
