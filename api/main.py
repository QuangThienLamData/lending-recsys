"""
api/main.py
------------
FastAPI application entry point.

Startup:
  Loads all heavyweight artefacts (embeddings, FAISS index, ranking model)
  into the `artefacts` dict once, before the first request is served.

Routes:
  GET  /health             — liveness probe
  POST /recommend          — main recommendation endpoint
  GET  /items              — list all known loan product types
  GET  /users/{user_id}    — check if a user exists in the training set
"""

import os
import json
import pickle
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env from project root (works regardless of working directory)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np
import pandas as pd
import faiss
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    RecommendRequest,
    RecommendResponse,
    HealthResponse,
)
from api.recommender import run_recommendation_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths (overrideable via environment variables) ────────────────────────────
MODEL_DIR         = os.getenv("MODEL_DIR",          "models/saved")
FAISS_INDEX_PATH  = os.getenv("FAISS_INDEX_PATH",   os.path.join(MODEL_DIR, "faiss.index"))
RANKING_MODEL_PATH= os.getenv("RANKING_MODEL_PATH", os.path.join(MODEL_DIR, "ranking_model.pt"))
ENCODERS_PATH     = os.getenv("ENCODERS_PATH",      os.path.join(MODEL_DIR, "encoders.pkl"))
META_PATH         = os.path.join("data", "processed", "feature_meta.json")
ITEM_LOOKUP_PATH        = os.path.join("data", "processed", "item_lookup.csv")
USER_PROFILE_PATH       = os.path.join("data", "processed", "user_profile_lookup.parquet")
USER_EMB_PATH     = os.path.join(MODEL_DIR, "als_user_embeddings.npy")
ITEM_EMB_PATH     = os.path.join(MODEL_DIR, "als_item_embeddings.npy")

# Global artefact store (populated at startup)
artefacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artefacts once at server startup; free them on shutdown."""
    logger.info("=== Loading artefacts into memory ===")

    # 1. Encoders (LabelEncoder for user_id and item_id)
    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    artefacts["user_enc"]         = enc["user_enc"]
    artefacts["item_enc"]         = enc["item_enc"]
    artefacts["user_transformer"] = enc.get("user_transformer")
    artefacts["item_transformer"] = enc.get("item_transformer")
    artefacts["top_states"]       = enc.get("top_states", [])
    logger.info("  Loaded encoders  (users=%d, items=%d)",
                len(artefacts["user_enc"].classes_),
                len(artefacts["item_enc"].classes_))

    # 2. ALS embeddings
    artefacts["user_emb"] = np.load(USER_EMB_PATH)
    artefacts["item_emb"] = np.load(ITEM_EMB_PATH)
    logger.info("  Loaded user_emb %s  item_emb %s",
                artefacts["user_emb"].shape, artefacts["item_emb"].shape)

    # 3. FAISS index
    artefacts["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
    logger.info("  Loaded FAISS index  ntotal=%d",
                artefacts["faiss_index"].ntotal)

    # 4. Ranking model predictor (also loads XGBoost repay predictor internally)
    from ranking.predictor import RankingPredictor
    artefacts["ranking_predictor"] = RankingPredictor(
        model_path=RANKING_MODEL_PATH,
        meta_path=META_PATH,
        device="cpu",
    )
    logger.info("  Loaded ranking model (%s)",
                artefacts["ranking_predictor"].model_type)
    if artefacts["ranking_predictor"].repay_predictor is not None:
        logger.info("  XGBoost repay predictor loaded (repay_feat_dim=%d)",
                    artefacts["ranking_predictor"].repay_feat_dim)
    else:
        logger.info("  XGBoost repay predictor not found — skipping repay stage")

    # 5. Item lookup table
    artefacts["item_lookup"] = pd.read_csv(ITEM_LOOKUP_PATH)
    logger.info("  Loaded item_lookup  (%d items)",
                len(artefacts["item_lookup"]))

    # 6. User profile lookup (demographics for all known users)
    if os.path.exists(USER_PROFILE_PATH):
        artefacts["user_profile_lookup"] = pd.read_parquet(USER_PROFILE_PATH)
        logger.info("  Loaded user_profile_lookup  (%d users)",
                    len(artefacts["user_profile_lookup"]))
    else:
        logger.warning("  user_profile_lookup not found — cold-start demographics unavailable")

    logger.info("=== All artefacts ready.  API is live. ===")
    yield

    # Cleanup
    artefacts.clear()
    logger.info("Artefacts cleared.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fintech Credit Product Recommender",
    description=(
        "End-to-end recommendation system for LendingClub loan products.\n\n"
        "Pipeline: ALS retrieval → FAISS ANN → NeuMF/DeepFM ranking → "
        "optional LLM rerank."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness probe — returns 200 if all artefacts are loaded."""
    return HealthResponse(
        status="ok",
        artefacts_loaded=list(artefacts.keys()),
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommend"])
def recommend(request: RecommendRequest):
    """
    Return top-K loan product recommendations for a borrower.

    - Retrieval: ALS user embedding → FAISS ANN search
    - Ranking:   NeuMF / DeepFM scores candidates
    - Optional:  LLM reranks using borrower profile + loan descriptions
    """
    if not artefacts:
        raise HTTPException(status_code=503, detail="Artefacts not yet loaded")

    result = run_recommendation_pipeline(request, artefacts)
    return JSONResponse(content=result)


@app.get("/items", tags=["Metadata"])
def list_items():
    """List all distinct loan product types known to the system."""
    if "item_lookup" not in artefacts:
        raise HTTPException(status_code=503, detail="item_lookup not loaded")
    return JSONResponse(
        content=artefacts["item_lookup"]
        .sort_values("n_loans", ascending=False)
        .to_dict(orient="records")
    )


@app.get("/users/{user_id}", tags=["Metadata"])
def user_info(user_id: str):
    """Check if a user_id exists in the training set."""
    if "user_enc" not in artefacts:
        raise HTTPException(status_code=503, detail="Encoders not loaded")
    known = user_id in set(artefacts["user_enc"].classes_)
    return {"user_id": user_id, "known": known,
            "cold_start": not known}
