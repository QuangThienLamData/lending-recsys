# Deployment Guide

> **Fintech Credit Product Recommendation System**
> ALS + FAISS Retrieval · DeepFM Ranking · FastAPI Serving

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Environment Setup — The Safe Way](#2-environment-setup--the-safe-way)
3. [Generating Model Artefacts](#3-generating-model-artefacts)
4. [Running the API Locally](#4-running-the-api-locally)
5. [Docker Deployment (Recommended)](#5-docker-deployment-recommended)
6. [Testing the API](#6-testing-the-api)

---

## 1. Project Architecture Overview

The system is split into two distinct phases: an **offline training pipeline** (run once in `notebooks/01_full_pipeline.ipynb`) that produces ALS embeddings, a FAISS vector index, and a trained DeepFM ranking model; and an **online serving layer** (FastAPI + uvicorn) that loads those frozen artefacts at startup and executes the full Retrieval → Ranking pipeline on every `/recommend` request.

```
┌─────────────────────────────────────────────────────────┐
│  OFFLINE  (Jupyter Notebook — run once locally)         │
│                                                         │
│  LendingClub CSV  →  Preprocessing  →  ALS Training     │
│                                         │               │
│                              FAISS Index + DeepFM Weights│
│                                         │               │
│                                  models/saved/  ◄───────┘
└─────────────────────────────────────────────────────────┘
                              │
                    volume-mount / COPY
                              │
┌─────────────────────────────────────────────────────────┐
│  ONLINE   (FastAPI server — runs continuously)          │
│                                                         │
│  POST /recommend                                        │
│    → FAISS retrieval  (top-50 candidates)               │
│    → DeepFM re-ranking                                  │
│    → JSON response  {rank, score, grade, purpose, …}    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Environment Setup — The Safe Way

### Why Conda and not plain pip?

Two packages in this project ship **pre-compiled C++ / Rust binaries** that are
notoriously fragile when installed through pip on Windows:

| Package | Binary dependency | Common symptom |
|---|---|---|
| `faiss-cpu` | C++ with OpenMP | `ImportError: DLL load failed` on Windows |
| `implicit` | C++ BLAS | circular import / segfault with numpy mismatch |
| `pydantic-core` (pulled in by FastAPI) | Rust | `mismatch between the Python and Rust versions` |

Installing these via **conda / conda-forge** resolves them against the same
ABI and avoids all three failure modes.

### Step 1 — Create the Conda environment

```bash
conda env create -f environment.yml
```

This resolves `faiss-cpu`, `pytorch`, `implicit`, and all data-science
dependencies from their conda-forge or pytorch channels before handing off
to pip for the API packages.

### Step 2 — Activate

```bash
conda activate credit_recsys
```

### Step 3 — Verify the install

```bash
python -c "import faiss, torch, implicit, fastapi; print('All imports OK')"
```

Expected output:
```
All imports OK
```

---

> ### ⚠️ Troubleshooting — `pydantic_core` Rust Binary Error
>
> **Symptom:** You see an error such as:
> ```
> ImportError: pydantic-core version mismatch — expected Rust ABI X, got Y
> ```
> or:
> ```
> TypeError: BaseModel.__init_subclass__() takes no keyword arguments
> ```
>
> **Cause:** `pydantic-core` was installed twice — once by conda and once by
> pip — resulting in a binary mismatch.
>
> **Fix:** Force-reinstall the entire FastAPI stack through pip with a clean
> cache, inside the active conda environment:
>
> ```bash
> conda activate credit_recsys
>
> pip install --force-reinstall --no-cache-dir \
>     fastapi>=0.110.0 \
>     uvicorn[standard]>=0.29.0 \
>     pydantic>=2.0.0 \
>     pydantic-core
> ```
>
> Then verify:
> ```bash
> python -c "from fastapi import FastAPI; print('FastAPI OK')"
> ```

---

### Updating the environment

If `environment.yml` changes after your initial setup:

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that were deleted from the spec.

---

## 3. Generating Model Artefacts

> **The FastAPI server will crash on startup if any of the files below are
> missing.** You must run the offline training notebook before starting the
> API for the first time.

### Step 1 — Download the dataset

Obtain the LendingClub dataset from Kaggle and place the CSV files under
`data/raw/`. You can use the Kaggle CLI (bundled in the conda environment):

```bash
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/raw/
```

### Step 2 — Run the offline training pipeline

Open and run **all cells** in:

```
notebooks/01_full_pipeline.ipynb
```

This notebook executes the full pipeline:
`Preprocessing → Sampling → ALS → FAISS → DeepFM Training → Evaluation`

Estimated runtime on a modern CPU: **15–40 minutes** depending on dataset size.

### Step 3 — Verify the required artefact files exist

After the notebook completes, confirm that **every file below is present**
before attempting to start the server:

```
models/saved/
├── encoders.pkl                ← LabelEncoders for user_id and item_id
├── faiss.index                 ← L2-normalised FAISS vector index
├── ranking_model.pt            ← Trained DeepFM state dict
├── als_user_embeddings.npy     ← ALS user embedding matrix  (n_users × 64)
└── als_item_embeddings.npy     ← ALS item embedding matrix  (n_items × 64)

data/processed/
├── feature_meta.json           ← n_users, n_items, model_type, feat dims
└── item_lookup.csv             ← item_idx → grade, purpose, term, int_rate …
```

**Windows (PowerShell):**
```powershell
Get-ChildItem models\saved
Get-ChildItem data\processed
```

**Linux / macOS:**
```bash
ls -lh models/saved/
ls -lh data/processed/
```

If any file is missing, re-run the notebook from the beginning.

---

## 4. Running the API Locally

> **⚠️ Critical — Working Directory**
>
> **Always run `uvicorn` from the project root directory**, not from inside
> `api/` or any subdirectory.
>
> The server resolves all artefact paths relative to the current working
> directory (e.g. `models/saved/faiss.index`). Running from the wrong folder
> causes an immediate `FileNotFoundError` on startup.
>
> **Correct:**
> ```
> C:\Users\spam8\Desktop\data\POC\recommendation system>  uvicorn api.main:app ...
> ```
> **Wrong:**
> ```
> C:\Users\spam8\Desktop\data\POC\recommendation system\api>  uvicorn main:app ...
> ```

### Start the server

```bash
# Make sure the conda environment is active first
conda activate credit_recsys

# Navigate to the project root
cd "C:\Users\spam8\Desktop\data\POC\recommendation system"   # Windows
# cd ~/path/to/recommendation\ system                         # Linux/macOS

# Start uvicorn from the project root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag restarts the server automatically when source files change.
Remove it in production.

### Expected startup log

```
INFO:root:=== Loading artefacts into memory ===
INFO:root:  Loaded encoders  (users=12437, items=196)
INFO:root:  Loaded user_emb (12437, 64)  item_emb (196, 64)
INFO:root:  Loaded FAISS index  ntotal=196
INFO:root:  Loaded ranking model (deepfm)
INFO:root:  Loaded item_lookup  (196 items)
INFO:root:=== All artefacts ready.  API is live. ===
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

If instead you see `FileNotFoundError: feature_meta.json not found`, you are
either in the wrong directory or the artefacts have not been generated yet.

---

## 5. Docker Deployment (Recommended)

Docker is the most reliable way to run the API — it eliminates all Windows /
macOS environment quirks, C++ binary conflicts, and working directory issues by
packaging the entire runtime into a single portable container.

### Prerequisites

- **Windows:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  must be installed, running (whale icon in the system tray), and set to
  **Linux containers** mode. WSL 2 must be enabled.
- **Linux / macOS:** Docker Engine 24+ with the Compose plugin.

Verify Docker is ready before proceeding:

```bash
docker version
# Both "Client" and "Server" sections must appear without error.
```

### Build and start

```bash
# From the project root
docker compose up --build
```

`--build` forces a fresh image build. Omit it on subsequent runs if source
code has not changed:

```bash
docker compose up          # reuse existing image, start the container
docker compose up -d       # same, but run in the background (detached)
docker compose logs -f     # follow logs from a detached container
```

### Verify the container is healthy

```bash
docker ps
```

The `STATUS` column should read **`Up X minutes (healthy)`** once the
artefacts have loaded (~10–20 seconds after the container starts).

### Stop the container

```bash
docker compose down
```

### How artefacts reach the container

The `docker-compose.yml` mounts your local artefact directories as read-only
volumes at runtime — so you can retrain the model locally and restart the
container without rebuilding the image:

```yaml
volumes:
  - ./models/saved:/app/models/saved:ro
  - ./data/processed:/app/data/processed:ro
```

To rebuild the image after source code changes:

```bash
docker compose up --build
```

---

## 6. Testing the API

Once the server is running (either locally or via Docker), use the commands
below to verify the full pipeline end-to-end.

### Liveness probe

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "artefacts_loaded": [
    "user_enc", "item_enc", "user_emb", "item_emb",
    "faiss_index", "ranking_predictor", "item_lookup"
  ]
}
```

### Get loan product recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "1234567",
    "top_k": 5,
    "retrieval_pool": 50
  }'
```

Expected response:
```json
{
  "user_id": "1234567",
  "n_returned": 5,
  "pipeline_stages": ["retrieval", "ranking"],
  "recommendations": [
    {
      "item_idx": 12,
      "item_id": "A_debt_consolidation_36 months",
      "grade": "A",
      "purpose": "debt_consolidation",
      "term": "36 months",
      "int_rate": 7.49,
      "loan_amnt": 14200.0,
      "positive_rate": 0.94,
      "rank": 1,
      "score": 0.8731
    }
  ]
}
```

### Cold-start user (not in training data)

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "BRAND_NEW_USER", "top_k": 3}'
```

The server returns the most popular loan products as a fallback —
no model inference is triggered for unknown users.

### List all known loan product types

```bash
curl http://localhost:8000/items
```

### Check whether a user exists in the training set

```bash
curl http://localhost:8000/users/1234567
```

### Interactive API documentation (browser)

The FastAPI server auto-generates a Swagger UI at:

```
http://localhost:8000/docs
```

All endpoints are explorable and executable directly from the browser —
useful for demos and presentations.

### PowerShell equivalent (Windows — no curl required)

```powershell
# Liveness probe
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Recommendation request
$body = '{"user_id": "1234567", "top_k": 5}'
Invoke-RestMethod -Uri "http://localhost:8000/recommend" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Run the unit test suite

```bash
# Conda environment active
pytest tests/ -v

# Inside the running Docker container
docker exec -it credit-recommender pytest tests/ -v
```

---

*Built as a bootcamp final project — PyTorch · FAISS · FastAPI · Docker*
