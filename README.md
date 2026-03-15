# Fintech Credit Product Recommendation System

End-to-end ML recommendation pipeline for LendingClub loan products.
Built as a bootcamp final project.

**Stack:** PyTorch · FAISS · FastAPI · Docker

---

## Design Choice: Synthetic Item Catalog
In this project, I engineered a "Synthetic Item Catalog" to adapt the LendingClub dataset for a collaborative filtering framework. Rather than treating every historical loan application as a unique item—which would result in millions of non-repeatable interactions—I defined a recommendable "product" by concatenating its core financial attributes: Grade, Purpose, and Term (e.g., B_debt_consolidation_36).

This approach intentionally mirrors a real-world retail banking scenario, where a financial institution recommends specific types of credit products to new borrowers, rather than past individual loans. Mathematically, this constrains the item space to a maximum of 196 distinct product configurations (7 grades × 14 purposes × 2 terms). While this highly concentrated item catalog makes Approximate Nearest Neighbor (ANN) search via FAISS computationally trivial for this specific dataset, the FAISS vector database was fully implemented to demonstrate mastery of a scalable, production-ready recommendation architecture.
## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download LendingClub data

```bash
bash scripts/download_data.sh
```

### 3. Run the full offline pipeline

```bash
bash scripts/run_pipeline.sh
```

This runs all 6 steps: preprocessing → ALS → FAISS → NeuMF/DeepFM → evaluation → ablation.

### 4. Start the API

```bash
# Docker (recommended)
docker compose up --build

# Or locally
uvicorn api.main:app --reload --port 8000
```

### 5. Make a recommendation request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1234567", "top_k": 5}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Pipeline Overview

```
Raw CSV → Preprocessing → ALS Training → FAISS Index → NeuMF/DeepFM → FastAPI
```

| Stage | Module | Output |
|---|---|---|
| Preprocessing | `preprocessing/` | `data/processed/*.npz` |
| ALS Training | `retrieval/train_als.py` | `models/saved/*_embeddings.npy` |
| FAISS Index | `retrieval/build_faiss_index.py` | `models/saved/faiss.index` |
| Ranking Model | `ranking/train_ranking.py` | `models/saved/ranking_model.pt` |
| API | `api/main.py` | REST endpoints on port 8000 |
| Evaluation | `evaluation/` | Recall@K, NDCG@K, ablation table |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/recommend` | Get top-K loan recommendations for a user |
| `GET`  | `/items` | List all known loan product types |
| `GET`  | `/users/{user_id}` | Check if a user is in the training set |

Interactive docs: http://localhost:8000/docs

---

## Project Structure

See [DESIGN.md](DESIGN.md) for the full architecture and data flow.

```
recommendation_system/
├── preprocessing/     Data loading, interaction matrix, feature engineering
├── models/            ALS (PyTorch), NeuMF, DeepFM model definitions
├── retrieval/         ALS training script + FAISS index builder
├── ranking/           Dataset, training loop, inference predictor
├── api/               FastAPI app, schemas, recommender logic, LLM reranker
├── evaluation/        Recall@K, NDCG@K, ablation study
├── tests/             pytest unit + integration tests
├── scripts/           run_pipeline.sh, download_data.sh
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
