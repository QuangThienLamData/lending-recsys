# Lending RecSys — Credit Product Recommendation System

An end-to-end, multi-stage ML recommendation pipeline for financial credit products,
built on the LendingClub dataset. The system combines approval prediction, collaborative
filtering, deep-feature-interaction ranking, and LLM re-ranking into a unified
**Streamlit web application** for both borrowers and internal staff.

**Stack:** Python · XGBoost · ALS · FAISS · DeepFM · OpenAI / Anthropic · Streamlit

---

## Problem Statement

This system is designed to solve four interconnected weaknesses that would exist if any
single component were missing:

### 1. Approval Prediction alone is not enough
An XGBoost model predicts the probability that a borrower will repay a loan. If this
were the **only** component, a rejected borrower simply hits a dead end — they leave
without completing their lending demand and the business loses a potential customer.

### 2. Recommendation alone is not enough
A pure recommendation system (without the approval prediction layer) would freely
suggest loan products to borrowers who have a low estimated repayment rate. Getting
approved for an unsuitable product can lead to default, which directly harms the
business's credit portfolio health.

### 3. Without LLM re-ranking, the RecSys is context-blind
A collaborative-filtering system only knows what past users like this borrower
accepted historically. It has no understanding of **why the borrower is borrowing
right now**. A user who says *"I want to buy a car worth $50,000"* should see
auto-loan products ranked first — not debt-consolidation loans that their profile
historically correlates with. The **LLM re-ranker** reads either the structured
form data or a free-text natural language prompt to reorder candidates by contextual
relevance.

### 4. Without an Internal Dashboard, business intelligence is lost
Every approval check is a sales opportunity. The **Internal Staff Dashboard** logs
each applicant query (with their profile, requested amount, and approval outcome)
so that staff can identify rejected customers for upsell and telesales follow-up.

---

## System Architecture

```
Borrower Input (Identity ID + Loan Request)
        │
        ▼
 Approval Prediction (XGBoost + IsotonicRegression Calibrator)
        │
        ├─ APPROVED ──────────────────────────────────────┐
        └─ REJECTED → show recommendations immediately    │
                                                          ▼
                    ALS Retrieval ← FAISS top-50 candidates
                          │
                          ▼
            Repayment Filter (≥65%) ← remove risky products
                          │
                          ▼
                    DeepFM Ranking ← top-10 candidates
                          │
                          ▼
               LLM Re-ranking (OpenAI / Anthropic)
               ├─ Prompt Mode: free-text borrower intent
               └─ Form Mode: structured loan request data
                          │
                          ▼
                    Final Top-5 Recommendations
                          │
                          ▼
               Internal Staff Dashboard (CSV log)
```

### Stage 1 — Retrieval (ALS + FAISS + Filter)
ALS factorises the implicit user×item interaction matrix into 64-dimensional embeddings.
Vectors are L2-normalised and indexed with `faiss.IndexFlatIP` for fast similarity
retrieval. **New Enhancement:** Immediately after retrieval, candidates are passed 
through the calibrated XGBoost repay predictor; only products with a **≥65% 
repayment probability** are passed to the ranking model. This ensures the recommendation
funnel is "safety-first".

### Stage 2 — Ranking (DeepFM)
A DeepFM model re-scores the FAISS candidate pool by jointly modelling second-order
feature interactions (FM layer) and higher-order interactions (MLP), trained on
popularity-weighted negative samples to reduce popularity bias.

### Stage 3 — LLM Re-ranking
The top-10 candidates are passed to OpenAI or Anthropic with a prompt encoding the
borrower's financial profile. In **AI Prompt Mode**, the borrower's free-text intent
(e.g. *"I want to buy a car worth $50,000"*) drives semantic reranking: the LLM
deduces the purpose, derives the required loan amount, and sorts candidates by
interest rate, term, and grade.

### Calibrated Approval Prediction
Raw XGBoost probabilities are post-processed through an **IsotonicRegression
calibrator** trained on the held-out validation split, yielding empirically
calibrated repayment probabilities. The approval threshold is **65%**.

---

## Key Results

| Stage | NDCG@5 | Recall@5 |
|-------|:------:|:--------:|
| Stage 1 — Retrieval Only | ~0.18 | ~0.14 |
| Stage 2 — + DeepFM Ranking | ~0.28 | ~0.22 |
| Stage 3 — + LLM Re-ranking | ~0.31 | ~0.25 |

---

## Quick Start

### Prerequisites

```bash
conda env create -f environment.yml
conda activate credit_recsys
# or
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...         # for OpenAI LLM re-ranking
# or
ANTHROPIC_API_KEY=sk-ant-...  # for Anthropic Claude
LLM_PROVIDER=openai           # "openai" or "anthropic"
```

### 1. Download LendingClub Data

```bash
bash scripts/download_data.sh   # requires ~/.kaggle/kaggle.json
```

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`.

### 2. Run the Training Pipeline

```bash
bash scripts/run_pipeline.sh
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `preprocessing/build_interactions.py` | interaction matrices |
| 2 | `preprocessing/feature_engineering.py` | feature arrays + encoders |
| 3 | `retrieval/train_als.py` | ALS embeddings |
| 4 | `retrieval/build_faiss_index.py` | FAISS index |
| 5 | `ranking/train_ranking.py` | DeepFM model |
| 6 | `ranking/train_xgboost.py` | XGBoost repay model |
| 7 | `ranking/train_calibrator.py` | IsotonicRegression calibrator |

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run the Ablation Study

```bash
python -m evaluation.ablation_study --k 5 --n-users 100 --pool 50
```

---

## Application Features

| Feature | Description |
|---------|-------------|
| **Quick Recommender** | Enter Identity ID + Name to instantly get loan recommendations |
| **Approval Check** | Full form submission with XGBoost + calibrated repayment score |
| **AI Prompt Mode** | Natural language prompt for LLM-driven contextual recommendations |
| **Rejection Flow** | Auto-shows recommendations + improvement tips when rejected |
| **SHAP Explanation** | Feature-level repayment score breakdown for each applicant |
| **Internal Dashboard** | Admin-only view of query history, charts, and acceptance metrics |

---

## Project Structure

```
credit_product_recommendation/
├── app.py                    Streamlit application (main entry point)
├── api/
│   ├── recommender.py        End-to-end recommendation pipeline
│   ├── explain.py            SHAP explanations + improvement suggestions
│   ├── llm_reranker.py       LLM re-ranking (OpenAI / Anthropic)
│   └── schemas.py            Pydantic request/response schemas
├── ranking/
│   ├── predictor.py          DeepFM inference wrapper
│   ├── repay_predictor.py    XGBoost + IsotonicRegression calibrator
│   ├── train_xgboost.py      XGBoost training script
│   └── train_calibrator.py   Calibration layer training script
├── retrieval/                ALS training + FAISS index builder
├── preprocessing/            Interaction matrix + feature engineering
├── evaluation/               NDCG@K, Recall@K, ablation study
├── generate_simulation_data.py  Seed script for internal dashboard history
├── data/                     (gitignored) raw + processed datasets
├── models/saved/             (gitignored) trained model artefacts
├── environment.yml
└── requirements.txt
```

---

## Internal Staff Dashboard

Access via the sidebar → **🔒 Internal Staff Dashboard** (credentials: `admin` / `admin`).

Tracks per-query: Identity ID, applicant name, requested amount, purpose, repayment
score, and approval outcome. Dashboard charts include query volume over time, purpose
breakdown, average repayment score by purpose, and overall acceptance rate.

## 🧪 Testing the Application

To verify the system's logic (Approval Prediction vs. Recommendations), you can use the following test cases in the **Loan Application** tab:

### 1. High Repayment Profile (Expected Result: APPROVED)
- **Identity ID**: `100001137`
- **Typical Input**: $10,000 | 36 Months | Debt Consolidation
- **Outcome**: This user has a high FICO and low DTI, resulting in a safe approval score.

### 2. High Risk Profiles (Expected Result: NOT APPROVED + AI Recommendations)
- **Identity ID**: `110727087` or `68499271`
- **Test Input**: $40,000 | 60 Months | Other
- **Outcome**: These IDs simulate profiles with lower repayment probabilities. The system will reject the $40,000 request but will immediately show **customized recommendation packages** that they *can* safely afford.

### 3. AI Personal Recommender (AI Prompt Mode)
Enable the **"Use AI Personal Loaning Recommendation"** checkbox and try a natural language prompt:
- **Sample Prompt**: *"I want to buy a car with price $20,000 but now I only have $10,000, which loaning package is the most suitable"*
- **Outcome**: The LLM will deduce the "Car" purpose, calculate the $10,000 gap, and prioritize Auto-loan products in the final ranking.
