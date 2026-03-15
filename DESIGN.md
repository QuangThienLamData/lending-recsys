# Fintech Credit Product Recommendation System
## Technical Design Document

**Project:** End-to-End Credit Product Recommender (Bootcamp Final Project)
**Dataset:** LendingClub (Kaggle)
**Stack:** PyTorch · FAISS · FastAPI · Docker

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Detailed Data Flow](#2-detailed-data-flow)
3. [Modular Codebase Structure](#3-modular-codebase-structure)
4. [Implementation Guide & Docker Strategy](#4-implementation-guide--docker-strategy)

---

## 1. System Architecture

The system is divided into two physically separate pipelines that share artefacts (saved model weights and FAISS index files) through a mounted volume or local file system.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                                 │
│                    (runs once, or on schedule)                          │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │  Raw CSVs    │───▶│  Preprocessing   │───▶│  Interaction Matrix  │  │
│  │  (LendingClub│    │  (feature eng.,  │    │  (user × item,       │  │
│  │   Kaggle)    │    │   deduplication, │    │   implicit ratings)  │  │
│  └──────────────┘    │   time split)    │    └──────────┬───────────┘  │
│                      └──────────────────┘               │              │
│                                                          │              │
│           ┌──────────────────────────────────────────────┘              │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │   ALS Training   │  (implicit-cf or custom PyTorch ALS)             │
│  │   user_emb.npy   │                                                   │
│  │   item_emb.npy   │                                                   │
│  └────────┬─────────┘                                                   │
│           │  item embeddings                                            │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  FAISS Indexing  │  (IndexFlatIP or IndexIVFFlat)                   │
│  │  faiss.index     │                                                   │
│  └──────────────────┘                                                   │
│                                                                         │
│           │  ALS embeddings + interaction data                         │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  NeuMF / DeepFM Training (PyTorch)               │  │
│  │  • Negative sampling against popularity                          │  │
│  │  • Input: (user_id, item_id, ALS emb, raw features)             │  │
│  │  • Output: relevance score ∈ [0, 1]                              │  │
│  │  • Saved: ranking_model.pt                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Artefacts written to:  models/saved/                                   │
│   ├── als_user_embeddings.npy                                           │
│   ├── als_item_embeddings.npy                                           │
│   ├── faiss.index                                                       │
│   ├── ranking_model.pt                                                  │
│   └── encoders.pkl   (label encoders, scaler, vocab)                   │
└─────────────────────────────────────────────────────────────────────────┘

                   shared volume / local file system
                   ────────────────────────────────▶

┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE PIPELINE                                  │
│                    (FastAPI server, always-on)                          │
│                                                                         │
│  Startup:                                                               │
│    load faiss.index  ──▶  RAM                                           │
│    load ranking_model.pt  ──▶  PyTorch (CPU/GPU)                       │
│    load als_*_embeddings.npy + encoders.pkl  ──▶  RAM                  │
│                                                                         │
│  Request Flow:                                                          │
│                                                                         │
│  HTTP POST /recommend                                                   │
│  { "user_id": "U123", "top_k": 10 }                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  Lookup user emb │  als_user_embeddings[user_id]                    │
│  └────────┬─────────┘                                                   │
│           │  user_vector (d,)                                           │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  FAISS Retrieval │  top-K*10 candidates via ANN (inner product)     │
│  └────────┬─────────┘                                                   │
│           │  candidate_item_ids  (K*10,)                               │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               NeuMF / DeepFM Ranking (PyTorch)                   │  │
│  │  • Build feature tensor for each (user, candidate) pair          │  │
│  │  • Forward pass → relevance scores                               │  │
│  │  • Sort descending → top-K                                       │  │
│  └────────┬─────────────────────────────────────────────────────────┘  │
│           │  ranked_item_ids  (K,)                                     │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  (Optional) LLM Reranker                                         │  │
│  │  • Compose prompt: borrower profile + top-K loan descriptions    │  │
│  │  • Call Claude / GPT-4 API                                       │  │
│  │  • Parse reranked list from response                             │  │
│  └────────┬─────────────────────────────────────────────────────────┘  │
│           │                                                             │
│           ▼                                                             │
│  HTTP 200  { "recommendations": [ {...loan details...} ] }             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Retrieval model | ALS (matrix factorisation) | Simple, interpretable, proven on implicit feedback |
| ANN library | FAISS `IndexFlatIP` (small) / `IndexIVFFlat` (larger) | Zero infra overhead, runs in-process |
| Ranking model | NeuMF (simpler) or DeepFM (richer feature interactions) | Reranks ALS candidates; corrects for popularity bias |
| API framework | FastAPI | Async, auto OpenAPI docs, easy Pydantic validation |
| Serialisation | `.npy` for embeddings, `torch.save` for model, `pickle` for encoders | Simple and portable for local/Docker use |

---

## 2. Detailed Data Flow

### 2.1 Data Preprocessing

#### Source Files
The LendingClub dataset contains one or more CSVs (e.g., `accepted_2007_to_2018Q4.csv`) with ~150 columns per loan application. Each row represents **one loan application by one borrower**.

#### Step 1 — Load & Filter

```
raw_df = pd.read_csv("data/raw/accepted_loans.csv", low_memory=False)

Keep columns:
  member_id          → user identifier (some versions use "id" or "emp_title" proxy)
  loan_amnt          → item feature
  term               → item feature  ("36 months" / "60 months")
  int_rate           → item feature
  grade / sub_grade  → item feature
  purpose            → item feature  (debt_consolidation, credit_card, home_improvement …)
  loan_status        → interaction signal
  issue_d            → timestamp for time-based split
  annual_inc         → user feature
  dti                → user feature  (debt-to-income ratio)
  fico_range_low/high→ user feature
  home_ownership     → user feature
  addr_state         → user feature (coarse geography)

Drop: free-text fields (desc, title) unless using LLM reranker
Drop: rows where member_id or loan_status is null
```

#### Step 2 — Define Users, Items, and Interactions

```
USER  = member_id  (one unique borrower)
ITEM  = a distinct loan product type, bucketed by (grade, purpose, term)
        → create a synthetic item_id:  item_id = grade + "_" + purpose + "_" + term
        e.g.  "B_debt_consolidation_36 months"  →  item_id = 42

INTERACTION SIGNAL (implicit):
  loan_status in {"Fully Paid", "Current"}   → rating = 1   (positive)
  loan_status in {"Charged Off", "Default"}  → rating = 0   (negative — keep for ranking)
  All others (Late, Grace Period)             → drop

Why implicit?
  Borrowers don't rate loans; their repayment behaviour IS the signal.
  ALS on implicit data is standard (Hu, Koren & Volinsky, 2008).
```

#### Step 3 — Encode IDs

```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder().fit(df["member_id"])
item_enc = LabelEncoder().fit(df["item_id"])

df["user_idx"] = user_enc.transform(df["member_id"])
df["item_idx"] = item_enc.transform(df["item_id"])

# Save encoders
import pickle
with open("models/saved/encoders.pkl", "wb") as f:
    pickle.dump({"user_enc": user_enc, "item_enc": item_enc}, f)
```

#### Step 4 — Time-Based Train / Val / Test Split

```
Sort all rows by issue_d ascending.

train : rows where issue_d < 2017-01-01   (~70%)
val   : rows where 2017-01-01 ≤ issue_d < 2018-01-01   (~15%)
test  : rows where issue_d ≥ 2018-01-01   (~15%)

This mimics production: the model is trained on historical data and
evaluated on future borrowers — preventing data leakage.

For each split, build a sparse interaction matrix:
  shape = (n_users, n_items),  value = 1 if positive interaction
```

#### Step 5 — Feature Engineering for Ranking Model

```
User features  (normalised with StandardScaler):
  annual_inc, dti, fico_range_low, fico_range_high,
  one-hot: home_ownership, addr_state (top 10 states + "other")

Item features  (from item_id lookup table):
  int_rate (float), loan_amnt_bucket (binned),
  one-hot: grade (A–G), purpose (14 categories), term (2 values)

Concatenate → feature_vector per (user, item) pair
Save scaler and one-hot vocabulary into encoders.pkl
```

---

### 2.2 Offline Training Phase

#### Stage A — ALS Training

ALS (Alternating Least Squares) factorises the implicit interaction matrix
`R ≈ U × Vᵀ` where `U ∈ ℝ^(n_users × d)` and `V ∈ ℝ^(n_items × d)`.

```
Option 1 (recommended for simplicity):
  pip install implicit
  from implicit.als import AlternatingLeastSquares

  model = AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
  model.fit(train_matrix.T)   # implicit expects item×user

Option 2 (PyTorch ALS — required if framework constraint is strict):
  Implement closed-form ALS update in PyTorch:
    U_i ← (VᵀV + λI)⁻¹ Vᵀ r_i
  Loop alternately over users and items for `n_iter` iterations.
  See  models/als_model.py

Export:
  np.save("models/saved/als_user_embeddings.npy", model.user_factors)
  np.save("models/saved/als_item_embeddings.npy", model.item_factors)
  Shapes: (n_users, 64) and (n_items, 64)
```

#### Stage B — Build FAISS Index

```python
import faiss, numpy as np

item_emb = np.load("models/saved/als_item_embeddings.npy").astype("float32")
d = item_emb.shape[1]   # 64

# Normalise for cosine similarity via inner product
faiss.normalize_L2(item_emb)

# For small datasets (< 100k items): exact search
index = faiss.IndexFlatIP(d)

# For larger datasets: approximate search
# quantiser = faiss.IndexFlatIP(d)
# index = faiss.IndexIVFFlat(quantiser, d, n_cells=100, metric=faiss.METRIC_INNER_PRODUCT)
# index.train(item_emb)

index.add(item_emb)
faiss.write_index(index, "models/saved/faiss.index")
print(f"FAISS index: {index.ntotal} items indexed")
```

#### Stage C — Negative Sampling & Ranking Model Training

```
Negative sampling strategy:
  For each (user, positive_item) in train set, sample k_neg=4 items
  that the user has NOT interacted with.
  Weight sampling by item popularity (popular negatives are harder).

  positive_label = 1
  negative_label = 0

Training dataset tensor:
  X = [user_idx, item_idx, user_features (F_u,), item_features (F_i,)]
  y = [0/1]

NeuMF Architecture (PyTorch):
  ┌──────────────────────────────────────────────────────────────────┐
  │  user_idx ──▶ Embedding(n_users, 32) ──┐                        │
  │                                         ├──▶ MLP([64,32,16,8])  │
  │  item_idx ──▶ Embedding(n_items, 32) ──┘         │              │
  │                                                    ▼             │
  │  user_idx ──▶ Embedding(n_users, 32) ──┐   concat([GMF, MLP])  │
  │                                         ├──▶ GMF (element-wise  │
  │  item_idx ──▶ Embedding(n_items, 32) ──┘   multiply)            │
  │                                                    │             │
  │                                              Linear(→1) + Sigmoid│
  └──────────────────────────────────────────────────────────────────┘

DeepFM Architecture (PyTorch — richer alternative):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Sparse features ──▶ Embedding lookup                            │
  │  Dense features  ──▶ normalised float                            │
  │                                                                  │
  │  FM Layer:  Σ_i Σ_j <v_i, v_j> x_i x_j  (2nd-order interactions)│
  │  Deep Layer: MLP([256, 128, 64])                                 │
  │  Output: FM_out + Deep_out ──▶ Sigmoid                           │
  └──────────────────────────────────────────────────────────────────┘

Training:
  Loss     = BCELoss
  Optimizer= Adam(lr=1e-3)
  Epochs   = 20 (early stopping on val NDCG@10)
  Batch    = 2048

Save:
  torch.save(model.state_dict(), "models/saved/ranking_model.pt")
```

---

### 2.3 Online Inference Phase

```
Step 1  Request arrives at  POST /recommend
        Body: { "user_id": "U123", "top_k": 10 }

Step 2  Encode user_id
        user_idx = user_enc.transform([user_id])[0]
        If unknown user → cold-start fallback (return top popular items)

Step 3  Retrieve user embedding
        user_vec = als_user_embeddings[user_idx]          # shape (64,)
        faiss.normalize_L2(user_vec.reshape(1,-1))

Step 4  FAISS ANN retrieval
        D, I = faiss_index.search(user_vec, k=top_k * 10)  # over-fetch
        candidate_item_idxs = I[0]                         # shape (K*10,)

Step 5  Build ranking feature tensor
        For each candidate item_idx:
          row = concat(user_features, item_features, als_user_emb, als_item_emb)
        tensor shape: (K*10, feature_dim)

Step 6  Ranking model forward pass
        model.eval()
        with torch.no_grad():
            scores = model(user_idx_tensor, candidate_tensor)   # (K*10,)
        top_k_idxs = scores.argsort(descending=True)[:top_k]
        top_k_item_idxs = candidate_item_idxs[top_k_idxs]

Step 7  (Optional) LLM Reranking
        prompt = build_rerank_prompt(user_profile, top_k_items)
        response = llm_client.chat(prompt)
        reranked_ids = parse_reranked_order(response, top_k_item_idxs)

Step 8  Decode and return
        item_names = item_enc.inverse_transform(reranked_ids)
        item_details = item_lookup_df.loc[item_names]
        return JSONResponse({"recommendations": item_details.to_dict("records")})
```

---

## 3. Modular Codebase Structure

```
recommendation_system/
│
├── data/
│   ├── raw/                          # Place downloaded LendingClub CSVs here (git-ignored)
│   │   └── accepted_2007_to_2018Q4.csv
│   ├── processed/                    # Output of preprocessing scripts
│   │   ├── train_interactions.npz    # Sparse user×item matrix, train split
│   │   ├── val_interactions.npz
│   │   ├── test_interactions.npz
│   │   ├── user_features.npy         # Scaled user feature matrix
│   │   ├── item_features.npy         # Item feature matrix
│   │   └── item_lookup.csv           # Mapping item_idx → loan attributes
│   └── __init__.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── build_interactions.py         # Loads raw CSV, defines user/item IDs, builds
│   │                                 # implicit interaction matrix and time-based splits.
│   ├── feature_engineering.py        # Constructs and normalises user and item feature
│   │                                 # vectors; fits and saves encoders.pkl + scaler.
│   └── negative_sampler.py           # Generates (user, neg_item) pairs for ranking
│                                     # training using popularity-weighted sampling.
│
├── models/
│   ├── __init__.py
│   ├── als_model.py                  # PyTorch implementation of ALS: closed-form user
│   │                                 # and item factor updates using torch.linalg.solve.
│   ├── neumf_model.py                # PyTorch NeuMF class: GMF branch + MLP branch
│   │                                 # fused at output layer with Sigmoid activation.
│   ├── deepfm_model.py               # PyTorch DeepFM class: factorisation machine layer
│   │                                 # for 2nd-order interactions + deep MLP branch.
│   └── saved/                        # Artefacts written by training scripts (git-ignored)
│       ├── als_user_embeddings.npy
│       ├── als_item_embeddings.npy
│       ├── faiss.index
│       ├── ranking_model.pt
│       └── encoders.pkl
│
├── retrieval/
│   ├── __init__.py
│   ├── train_als.py                  # Orchestrates ALS training (calls als_model.py or
│   │                                 # implicit library), exports .npy embedding files.
│   └── build_faiss_index.py          # Loads item embeddings, normalises them, builds
│                                     # FAISS index (Flat or IVF), and writes faiss.index.
│
├── ranking/
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch Dataset class: reads interaction data +
│   │                                 # negative samples, returns (user_idx, item_idx,
│   │                                 # features, label) tensors.
│   ├── train_ranking.py              # Training loop for NeuMF/DeepFM: BCELoss, Adam,
│   │                                 # early stopping on val NDCG@10, saves best checkpoint.
│   └── predictor.py                  # Inference wrapper: loads ranking_model.pt, exposes
│                                     # score_candidates(user_idx, candidate_item_idxs).
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point: loads all artefacts at
│   │                                 # startup via @app.on_event("startup"), defines routes.
│   ├── schemas.py                    # Pydantic models: RecommendRequest, RecommendResponse,
│   │                                 # ItemDetail — used for request validation & OpenAPI docs.
│   ├── recommender.py                # Core recommendation logic: FAISS retrieval →
│   │                                 # ranking model → optional LLM reranker pipeline.
│   └── llm_reranker.py               # Optional: builds prompt from borrower profile and
│                                     # candidate loan descriptions; calls LLM API; parses
│                                     # reranked list from structured response.
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                    # Implements Recall@K and NDCG@K as pure Python/NumPy
│   │                                 # functions operating on ranked item lists.
│   ├── evaluate_pipeline.py          # Runs full evaluation on the test split: iterates
│   │                                 # users, calls each pipeline stage, records metrics.
│   └── ablation_study.py             # Compares three configurations side-by-side:
│                                     # (1) Retrieval only, (2) Retrieval+Ranking,
│                                     # (3) Retrieval+Ranking+LLM. Outputs a summary table.
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis: distribution of loan
│   │                                 # grades, purposes, FICO scores, interaction sparsity.
│   ├── 02_ALS_Experiment.ipynb       # Interactive ALS training with embedding visualisation
│   │                                 # (PCA/UMAP of item embeddings coloured by grade).
│   └── 03_Ranking_Experiment.ipynb   # NeuMF/DeepFM training curves and val metric plots.
│
├── scripts/
│   ├── run_pipeline.sh               # End-to-end bash script: preprocess → train_als →
│   │                                 # build_faiss → train_ranking → evaluate.
│   └── download_data.sh              # Kaggle API download helper (requires kaggle.json).
│
├── tests/
│   ├── test_metrics.py               # Unit tests for Recall@K and NDCG@K correctness.
│   ├── test_neumf.py                 # Smoke test: forward pass through NeuMF with random
│   │                                 # tensors, asserts output shape and value range.
│   └── test_api.py                   # FastAPI TestClient integration test: POST /recommend
│                                     # with a known user_id, assert top-K in response.
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── DESIGN.md                         # This document
└── README.md
```

---

## 4. Implementation Guide & Docker Strategy

### 4.1 Local Setup (no Docker)

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate                           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (requires Kaggle API key at ~/.kaggle/kaggle.json)
bash scripts/download_data.sh

# 4. Run the full offline pipeline
bash scripts/run_pipeline.sh

# 5. Start the API server
uvicorn api.main:app --reload --port 8000
```

**`requirements.txt`**
```
torch>=2.0.0
faiss-cpu>=1.7.4
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
implicit>=0.7.2
scipy>=1.11.0
httpx>=0.27.0       # for TestClient in tests
openai>=1.0.0       # optional, for LLM reranker
```

---

### 4.2 Dockerfile

The Dockerfile covers **only the online (serving) phase**. Offline training is run locally and artefacts are copied into the image (or mounted at runtime).

```dockerfile
# Dockerfile
FROM python:3.11-slim

# --- System deps (FAISS needs libgomp) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (layer-cached) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY api/        ./api/
COPY models/     ./models/
COPY ranking/    ./ranking/
COPY retrieval/  ./retrieval/
COPY preprocessing/ ./preprocessing/

# --- Pre-trained artefacts ---
#     Option A: bake them into the image (simple, larger image)
COPY models/saved/ ./models/saved/

#     Option B: mount them at runtime via docker-compose volume (preferred)
#     (leave this COPY commented out and use the volume in docker-compose.yml)

EXPOSE 8000

# Uvicorn starts the FastAPI app; artefacts are loaded inside startup event
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 4.3 docker-compose.yml

```yaml
# docker-compose.yml
version: "3.9"

services:

  recommender-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: credit-recommender
    ports:
      - "8000:8000"
    volumes:
      # Mount pre-trained artefacts from host without rebuilding the image
      - ./models/saved:/app/models/saved:ro
    environment:
      - MODEL_DIR=/app/models/saved
      - FAISS_INDEX_PATH=/app/models/saved/faiss.index
      - RANKING_MODEL_PATH=/app/models/saved/ranking_model.pt
      - ENCODERS_PATH=/app/models/saved/encoders.pkl
      # Optional LLM reranker key (never hardcode secrets)
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

---

### 4.4 FastAPI Startup — Loading Artefacts into Memory

```python
# api/main.py
import os
import pickle
import numpy as np
import faiss
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

from models.neumf_model import NeuMF
from api.schemas import RecommendRequest, RecommendResponse

# Global artefact store (loaded once at startup)
artefacts: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy artefacts into memory before serving requests."""
    model_dir = os.getenv("MODEL_DIR", "models/saved")

    # 1. Encoders
    with open(f"{model_dir}/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    artefacts["user_enc"] = enc["user_enc"]
    artefacts["item_enc"] = enc["item_enc"]

    # 2. ALS embeddings
    artefacts["user_emb"] = np.load(f"{model_dir}/als_user_embeddings.npy")
    artefacts["item_emb"] = np.load(f"{model_dir}/als_item_embeddings.npy")

    # 3. FAISS index
    artefacts["faiss_index"] = faiss.read_index(f"{model_dir}/faiss.index")

    # 4. Ranking model
    n_users = artefacts["user_emb"].shape[0]
    n_items = artefacts["item_emb"].shape[0]
    model = NeuMF(n_users=n_users, n_items=n_items, emb_dim=32)
    model.load_state_dict(torch.load(f"{model_dir}/ranking_model.pt", map_location="cpu"))
    model.eval()
    artefacts["ranking_model"] = model

    print("All artefacts loaded. API is ready.")
    yield
    # Cleanup (optional)
    artefacts.clear()


app = FastAPI(title="Credit Recommender API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "artefacts_loaded": list(artefacts.keys())}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    from api.recommender import run_recommendation_pipeline
    return run_recommendation_pipeline(req, artefacts)
```

---

### 4.5 Deployment Commands

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up --build -d

# View logs
docker compose logs -f recommender-api

# Test the endpoint
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U123", "top_k": 5}'

# Stop
docker compose down
```

---

## 5. Evaluation Protocol

### Metrics

```python
# evaluation/metrics.py

def recall_at_k(recommended: list, ground_truth: set, k: int) -> float:
    """Fraction of ground-truth items captured in top-K recommendations."""
    hits = len(set(recommended[:k]) & ground_truth)
    return hits / min(len(ground_truth), k)

def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.
    Rewards relevant items appearing earlier in the ranked list.
    """
    import math
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in ground_truth
    )
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
```

### Ablation Study Design

| Configuration | Retrieval | Ranking | LLM Rerank | Expected Recall@10 | Expected NDCG@10 |
|---|---|---|---|---|---|
| Baseline (popular items) | ✗ | ✗ | ✗ | ~0.05 | ~0.04 |
| Retrieval only (ALS+FAISS) | ✓ | ✗ | ✗ | ~0.25 | ~0.18 |
| Retrieval + Ranking | ✓ | ✓ | ✗ | ~0.35 | ~0.28 |
| Full pipeline (+LLM) | ✓ | ✓ | ✓ | ~0.38 | ~0.32 |

> **Note:** Exact numbers depend on dataset size, hyperparameters, and negative sampling strategy. Use the ablation to demonstrate relative improvement, not absolute benchmarks.

```bash
# Run full ablation
python evaluation/ablation_study.py \
  --test-data data/processed/test_interactions.npz \
  --k 10 \
  --output results/ablation_results.csv
```

---

## 6. End-to-End Pipeline Execution Order

```
1.  python preprocessing/build_interactions.py
2.  python preprocessing/feature_engineering.py
3.  python retrieval/train_als.py
4.  python retrieval/build_faiss_index.py
5.  python ranking/train_ranking.py
6.  python evaluation/evaluate_pipeline.py
7.  python evaluation/ablation_study.py
8.  docker compose up --build          # serve the API
```

Each script is independently runnable, reads from `data/processed/` or `models/saved/`, and writes its outputs there. This makes the pipeline easy to re-run partially (e.g., retrain only the ranking model without rerunning ALS).

---

*Document generated: 2026-03-13*
*Stack versions: Python 3.11, PyTorch 2.x, FAISS 1.7.x, FastAPI 0.110.x*
