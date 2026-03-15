# Pipeline Run Guide

Step-by-step instructions for running the full offline ML pipeline.
All commands must be run from the **project root** (`recommendation system/`).

---

## Environment

Always activate the conda env before running anything:

```bash
conda activate credit_recsys
```

If conda isn't on your PATH (common on Windows), use the full Python path instead:

```
C:\Users\spam8\.conda\envs\credit_recsys\python.exe
```

All examples below use `python` — replace with the full path if needed.

---

## Artefact Status (as of 2026-03-15)

| Artefact | Location | Status |
|---|---|---|
| `accepted_2007_to_2018Q4.csv` | `data/raw/` | ✅ Exists |
| `train/val/test_interactions.npz` | `data/processed/` | ✅ Generated |
| `interactions_all.parquet` | `data/processed/` | ✅ Generated |
| `user_id_map.csv`, `item_lookup.csv` | `data/processed/` | ✅ Generated |
| `user_features.npy` (2,223,667 × 21) | `data/processed/` | ✅ Generated |
| `item_features.npy` (194 × 25) | `data/processed/` | ✅ Generated |
| `feature_meta.json` | `data/processed/` | ✅ Updated |
| `neg_samples_train.npy` | `data/processed/` | ✅ Generated |
| `als_user_embeddings.npy` (2,223,667 × 64) | `models/saved/` | ✅ Generated |
| `als_item_embeddings.npy` (194 × 64) | `models/saved/` | ✅ Generated |
| `faiss.index` (IndexFlatIP, 194 items) | `models/saved/` | ✅ Generated |
| `encoders.pkl` | `models/saved/` | ✅ Generated |
| `ranking_model.pt` | `models/saved/` | ✅ Generated (best epoch 1, NDCG@10=0.3532) |

---

## Step 1 — Preprocessing: Build Interaction Matrices

**Script:** `preprocessing/build_interactions.py`

```bash
python -m preprocessing.build_interactions
```

**What it does:**
- Reads `data/raw/accepted_2007_to_2018Q4.csv` (2.26M rows)
- Uses `id` as user proxy (member_id column is fully null in this dataset)
- Creates synthetic item IDs: `grade_purpose_term` → 194 unique products
- Splits: 60% train / 20% val / 20% test (chronological)
- Saves: `train_interactions.npz`, `val_interactions.npz`, `test_interactions.npz`, `interactions_all.parquet`, `item_lookup.csv`, `user_id_map.csv`, `encoders.pkl`

**Output dimensions:** 2,223,667 users × 194 items

> ✅ **Already done** — all output files exist in `data/processed/`

---

## Step 2 — Preprocessing: Feature Engineering

**Script:** `preprocessing/feature_engineering.py`

```bash
python -m preprocessing.feature_engineering
```

**What it does:**
- Reads `interactions_all.parquet`
- Builds user feature matrix: 21 numeric features per user (loan amount, rate, income, etc.)
- Builds item feature matrix: 25 features per item (grade OHE, purpose OHE, term)
- Updates `models/saved/encoders.pkl` with fitted scalers
- Saves: `user_features.npy` (2,223,667 × 21), `item_features.npy` (194 × 25), `feature_meta.json`

> ✅ **Already done** — output files exist in `data/processed/`

---

## Step 3 — Preprocessing: Negative Sampling

**Script:** `preprocessing/negative_sampler.py`

```bash
python -m preprocessing.negative_sampler --k-neg 4
```

**What it does:**
- For each positive (user, item) interaction (1,088,440 pairs), samples 4 negatives
- Popularity-weighted sampling: items the user never saw, weighted by item frequency^0.75
- Saves: `neg_samples_train.npy` (shape: ~4.35M × 2, dtype int32)

**Expected runtime:** ~50 seconds

> ✅ **Already done** — `neg_samples_train.npy` (34 MB) exists in `data/processed/`

---

## Step 4 — Retrieval: Train ALS

**Script:** `retrieval/train_als.py`

```bash
python -m retrieval.train_als --factors 64 --iterations 20 --backend implicit
```

**What it does:**
- Loads `data/processed/train_interactions.npz` (2,223,667 × 194)
- Trains Alternating Least Squares via the `implicit` C++ library
- Saves: `models/saved/als_user_embeddings.npy` (2,223,667 × 64), `models/saved/als_item_embeddings.npy` (194 × 64)

**Expected runtime:** ~55 seconds

> ✅ **Already done** — fresh embeddings exist in `models/saved/`

---

## Step 5 — Retrieval: Build FAISS Index

**Script:** `retrieval/build_faiss_index.py`

```bash
python -m retrieval.build_faiss_index --index-type auto
```

**What it does:**
- Loads `models/saved/als_item_embeddings.npy` (194 × 64)
- L2-normalises embeddings (so inner product = cosine similarity)
- Builds `IndexFlatIP` (exact search, chosen because n_items=194 < 50,000 threshold)
- Saves: `models/saved/faiss.index`

**Expected runtime:** < 2 seconds

> ✅ **Already done** — `models/saved/faiss.index` exists (194 vectors, d=64)

---

## Step 6 — Ranking: Train DeepFM

**Script:** `ranking/train_ranking.py`

```bash
python -m ranking.train_ranking --model deepfm --epochs 20 --batch 2048 --patience 5 --device cpu
```

**What it does:**
- Loads `RankingDataset`: 1,088,440 positives + 4,353,760 negatives = **5,442,200 training samples**
- Loads side features: user_features (2,223,667 × 21) + item_features (194 × 25)
- Builds DeepFM: sparse fields [n_users=2,223,667, n_items=194], dense_dim=46, emb_dim=16, MLP=[256,128,64]
- **37.8M parameters** (embedding table dominates: 2.2M × 16 = 35.5M)
- Trains with BCELoss + Adam + ReduceLROnPlateau + early stopping on val NDCG@10
- Saves best checkpoint to `models/saved/ranking_model.pt`

> ⚠️ **GPU NOTE:** RTX 5060 Ti (sm_120) is **not supported** by PyTorch ≤ 2.4 (max sm_90).
> Always pass `--device cpu` explicitly. PyTorch will default to CUDA even with an incompatible GPU.
> To use the GPU, upgrade PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu128`

**Expected runtime on CPU:** ~15–30 minutes per epoch depending on CPU. With early stopping (patience=5), likely stops at epoch 5–8. Total: **1–4 hours**.

To train faster, use NeuMF (fewer parameters):
```bash
python -m ranking.train_ranking --model neumf --epochs 20 --batch 2048 --patience 5 --device cpu
```

> ✅ **Done** — `models/saved/ranking_model.pt` saved at epoch 1 (val NDCG@10 = **0.3532**).
>
> **Overfitting note:** This dataset has ~1 interaction per user, making generalisation hard. Training ran 6 epochs before early stopping; NDCG peaked at epoch 1 and dropped to 0.05 by epoch 6. The best-checkpoint restore means the saved model is always from the peak epoch.
>
> | Epoch | Train Loss | Val NDCG@10 |
> |---|---|---|
> | 1 | 0.4485 | **0.3532** ← saved |
> | 2 | 0.3517 | 0.1662 |
> | 3 | 0.1328 | 0.1025 |
> | 4 | 0.0421 | 0.0849 |
> | 5 | 0.0152 | 0.0645 |
> | 6 | 0.0083 | 0.0519 |
> | — | — | early stop |

---

## Step 7 — Evaluation: Full Pipeline

**Script:** `evaluation/evaluate_pipeline.py`

```bash
python -m evaluation.evaluate_pipeline --k 10 --n-users 2000 --pool 100
```

**What it does:**
- For each of 2,000 test users:
  1. FAISS retrieval: top-100 candidates via ALS cosine similarity
  2. DeepFM re-ranking: score all 100 candidates, take top-10
  3. Compare against test-set ground truth
- Reports: Recall@10, Precision@10, NDCG@10, HitRate@10, MRR
- Saves: `evaluation/results.json`

> ✅ `ranking_model.pt` exists — you can run this now.

---

## Step 8 — Evaluation: Ablation Study

**Script:** `evaluation/ablation_study.py`

```bash
python -m evaluation.ablation_study --k 10 --n-users 500 --pool 100
```

**What it does:**
- Compares 3 configurations on 500 test users:
  - **Config 1:** ALS + FAISS only (no ranking model)
  - **Config 2:** ALS + FAISS + DeepFM/NeuMF re-ranking
  - **Config 3:** Config 2 + LLM reranker (skipped if no API key set)
- Prints Markdown table to stdout
- Saves: `evaluation/ablation_results.csv`

> ✅ `ranking_model.pt` exists — you can run this now.

---

## Run Everything at Once

```bash
bash scripts/run_pipeline.sh --model deepfm --epochs 20
```

The shell script passes `--device` only to the ranking step. Edit `scripts/run_pipeline.sh` to add `--device cpu` to the ranking step if GPU is not compatible:

```bash
# In run_pipeline.sh, find the [6/6] block and change to:
python -m ranking.train_ranking \
  --model "$RANK_MODEL" \
  --epochs "$RANK_EPOCHS" \
  --device cpu
```

---

## Start the API (after training)

```bash
# Local
uvicorn api.main:app --reload --port 8000

# Docker
docker compose up --build
```

Test it:
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1234567", "top_k": 5}'
```

---

## Run Tests

```bash
pytest tests/ -v
```

Tests are fully mock-based (`unittest.mock`) — no trained model required.

---

## Bugs Fixed During This Run

| File | Bug | Fix |
|---|---|---|
| `preprocessing/negative_sampler.py` | `weights.sum()` ≠ 1.0 after `astype(float64)` — float32→float64 precision drift causes `rng.choice` ValueError | Normalize **after** cast |
| `retrieval/train_als.py` | `model.user_factors` and `model.item_factors` were swapped — implicit's "user_factors" are the row factors of the item×user matrix (i.e., item factors) | Swapped return: `model.item_factors, model.user_factors` |
| `ranking/train_ranking.py` | `from dataset import RankingDataset` fails as module import | Changed to `from ranking.dataset import RankingDataset` |
| `ranking/train_ranking.py` | Ternary operator precedence bug: `"cuda" if ... else "cpu" if device_str=="auto" else device_str` always picks CUDA | Replaced with explicit `if/else` block |
| `ranking/train_ranking.py` | `evaluate()` called `model(u, i, uf, itf)` with 4 args for DeepFM which only accepts 2 | Added `model_type` param to `evaluate()` with proper dispatch |
| `models/deepfm_model.py` | `build_deepfm` defined 5 sparse fields `[n_users, n_items, n_grades, n_purposes, n_terms]` but training/inference only pass 2 (`[user_idx, item_idx]`) — would crash at `sparse_inputs[:, 2]` | Reduced to `[n_users, n_items]` |
| `data/processed/feature_meta.json` | `model_type: "DeepFM"` (capital D) — predictor.py string-compares against `"neumf"` | Changed to lowercase `"deepfm"`, added `user_feat_dim`/`item_feat_dim` keys |

---

## Key Data Facts

| Fact | Value |
|---|---|
| Raw rows | 2,260,701 |
| Users (unique loan IDs) | 2,223,667 |
| Items (grade × purpose × term) | 194 |
| Train interactions | 1,088,440 |
| Val interactions | 431,172 |
| Test interactions | 483,499 |
| Negative training samples | ~4,353,760 (4× positives) |
| ALS embedding dim | 64 |
| DeepFM parameters | 37,867,975 |
