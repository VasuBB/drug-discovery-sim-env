FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# RDKit's wheels need a couple of system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/server/requirements.txt

# Copy the full repo (models.py at root + server package)
COPY . /app

ENV PORT=8000 HOST=0.0.0.0 WORKERS=1
EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
