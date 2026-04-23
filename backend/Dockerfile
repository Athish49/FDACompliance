# FDA Compliance AI — FastAPI backend (Render / generic container hosts)
# Build context: this directory (`backend/`) as root, per Render "Root Directory".

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Native deps for scientific Python wheels and tokenizers when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU PyTorch first so FlagEmbedding does not pull a full CUDA stack
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# App code only — `data/` (CFR XML / chunk JSON) is excluded via `.dockerignore`;
# cloud deploys use Qdrant Cloud + `QDRANT_*` env; no ingestion runs on container start.
COPY . .

# Default for local `docker run -p 8000:8000`. Render sets PORT at runtime.
EXPOSE 8000

# Render injects $PORT; local/dev can omit it
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
