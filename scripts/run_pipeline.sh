#!/usr/bin/env bash
# scripts/run_pipeline.sh
# ---------------------------------------------------------------------------
# End-to-end offline pipeline runner.
# Run this once after downloading the LendingClub CSV.
#
# Usage:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --model deepfm --factors 128
#
# Options (passed through to the relevant Python scripts):
#   --raw        Path to raw CSV  (default: data/raw/accepted_2007_to_2018Q4.csv)
#   --factors    ALS embedding dim (default: 64)
#   --iterations ALS iterations   (default: 20)
#   --backend    ALS backend: implicit | pytorch (default: implicit)
#   --model      Ranking model: neumf | deepfm (default: neumf)
#   --epochs     Ranking training epochs (default: 20)
#   --k          Evaluation cutoff k  (default: 10)
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
RAW_CSV="data/raw/accepted_2007_to_2018Q4.csv"
ALS_FACTORS=64
ALS_ITER=20
ALS_BACKEND="implicit"
RANK_MODEL="neumf"
RANK_EPOCHS=20
EVAL_K=10

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw)        RAW_CSV="$2";       shift 2 ;;
    --factors)    ALS_FACTORS="$2";   shift 2 ;;
    --iterations) ALS_ITER="$2";      shift 2 ;;
    --backend)    ALS_BACKEND="$2";   shift 2 ;;
    --model)      RANK_MODEL="$2";    shift 2 ;;
    --epochs)     RANK_EPOCHS="$2";   shift 2 ;;
    --k)          EVAL_K="$2";        shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo " Fintech Credit Recommender — Offline Pipeline"
echo "============================================================"
echo " Raw CSV   : $RAW_CSV"
echo " ALS       : factors=$ALS_FACTORS  iter=$ALS_ITER  backend=$ALS_BACKEND"
echo " Ranking   : model=$RANK_MODEL  epochs=$RANK_EPOCHS"
echo " Eval k    : $EVAL_K"
echo "============================================================"

# ── Step 1: Build interaction matrices ───────────────────────────────────────
echo ""
echo "[1/6] Building interaction matrices …"
python -m preprocessing.build_interactions --raw "$RAW_CSV"

# ── Step 2: Feature engineering ──────────────────────────────────────────────
echo ""
echo "[2/6] Engineering features …"
python -m preprocessing.feature_engineering

# ── Step 3: Generate negative samples ────────────────────────────────────────
echo ""
echo "[3/6] Generating negative samples …"
python -m preprocessing.negative_sampler --k-neg 4

# ── Step 4: Train ALS ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Training ALS …"
python -m retrieval.train_als \
  --factors "$ALS_FACTORS" \
  --iterations "$ALS_ITER" \
  --backend "$ALS_BACKEND"

# ── Step 5: Build FAISS index ─────────────────────────────────────────────────
echo ""
echo "[5/6] Building FAISS index …"
python -m retrieval.build_faiss_index --index-type auto

# ── Step 6: Train ranking model ───────────────────────────────────────────────
echo ""
echo "[6/6] Training ranking model ($RANK_MODEL) …"
python -m ranking.train_ranking \
  --model "$RANK_MODEL" \
  --epochs "$RANK_EPOCHS"

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "[eval] Evaluating full pipeline on test set …"
python -m evaluation.evaluate_pipeline --k "$EVAL_K"

echo ""
echo "[ablation] Running ablation study …"
python -m evaluation.ablation_study --k "$EVAL_K" --n-users 500

echo ""
echo "============================================================"
echo " Pipeline complete!  Artefacts saved to models/saved/"
echo " Start the API with:  docker compose up --build"
echo " or locally with:     uvicorn api.main:app --reload --port 8000"
echo "============================================================"
