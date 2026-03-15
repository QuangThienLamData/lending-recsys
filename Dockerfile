# ── Fintech Credit Product Recommender — Serving Image ──────────────────────
#
# This Dockerfile packages ONLY the online inference layer (FastAPI server).
# Offline training is run locally first; artefacts are mounted via docker-compose.
#
# Build:  docker build -t credit-recommender .
# Run:    docker compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# libgomp1  : required by FAISS CPU build for OpenMP parallelism
# curl      : used by the Docker health-check command
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached as a separate layer) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────────────────────
COPY api/            ./api/
COPY models/         ./models/
COPY ranking/        ./ranking/
COPY retrieval/      ./retrieval/
COPY preprocessing/  ./preprocessing/
COPY evaluation/     ./evaluation/
COPY data/processed/ ./data/processed/

# ── Pre-trained artefacts ─────────────────────────────────────────────────────
# Option A (default): copy artefacts into the image for self-contained deployment
# COPY models/saved/ ./models/saved/
#
# Option B (recommended for iterative development):
# Leave this commented out and mount models/saved/ via docker-compose volume.
# The compose file below uses Option B so you can retrain without rebuilding.

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Uvicorn with 2 workers; switch to --workers 4 on multi-core production hosts
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
