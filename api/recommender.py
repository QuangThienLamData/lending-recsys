"""
api/recommender.py
-------------------
Core recommendation pipeline called by api/main.py for each request.

Pipeline:
  1. Encode user_id → user_idx  (cold-start fallback if unknown)
  2. Lookup ALS user embedding
  3. FAISS ANN retrieval  → candidate item indices
  4. RankingPredictor     → relevance scores for candidates
  5. (Optional) LLM rerank
  6. Decode item indices → item details
  7. Build RecommendResponse
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ITEM_LOOKUP_PATH = os.path.join("data", "processed", "item_lookup.csv")


# ── Popular items fallback (cold start) ──────────────────────────────────────

def _get_popular_items(item_lookup: pd.DataFrame, top_k: int) -> np.ndarray:
    """Return indices of the most popular items (by n_loans) for cold-start users."""
    return (
        item_lookup.sort_values("n_loans", ascending=False)
        .head(top_k)["item_idx"]
        .values.astype(np.int64)
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_recommendation_pipeline(request, artefacts: dict) -> dict:
    """
    Parameters
    ----------
    request     : RecommendRequest  (Pydantic model from api/schemas.py)
    artefacts   : dict loaded in api/main.py startup
                  keys: user_enc, item_enc, user_emb, item_emb,
                        faiss_index, ranking_model_predictor, item_lookup

    Returns
    -------
    dict matching RecommendResponse schema
    """
    import faiss

    user_enc      = artefacts["user_enc"]
    item_lookup   = artefacts["item_lookup"]
    user_emb      = artefacts["user_emb"]       # (n_users, d)
    faiss_index   = artefacts["faiss_index"]
    predictor     = artefacts["ranking_predictor"]

    pipeline_stages = ["retrieval", "ranking"]
    top_k    = request.top_k
    pool     = request.retrieval_pool

    # ── Step 1: Encode user ──────────────────────────────────────────────
    user_id = request.user_id
    known_users = set(user_enc.classes_)
    is_cold_start = user_id not in known_users

    if is_cold_start:
        logger.info("Cold-start user '%s' — returning popular items.", user_id)
        candidate_idxs = _get_popular_items(item_lookup, pool)
        user_vec       = None
    else:
        user_idx = int(user_enc.transform([user_id])[0])
        user_vec = user_emb[user_idx].astype("float32").reshape(1, -1)
        faiss.normalize_L2(user_vec)

        # ── Step 2: FAISS retrieval ──────────────────────────────────────
        _, I = faiss_index.search(user_vec, pool)
        candidate_idxs = I[0]

    # ── Step 3: Ranking model ────────────────────────────────────────────
    if not is_cold_start:
        scores = predictor.score_candidates(user_idx, candidate_idxs)
        sorted_order = np.argsort(scores)[::-1][:top_k]
        top_item_idxs = candidate_idxs[sorted_order]
        top_scores    = scores[sorted_order]
    else:
        top_item_idxs = candidate_idxs[:top_k]
        top_scores    = np.zeros(len(top_item_idxs), dtype=np.float32)

    # ── Step 4: Build candidates DataFrame ──────────────────────────────
    idx_to_row = item_lookup.set_index("item_idx")
    rows = []
    for rank, (idx, score) in enumerate(zip(top_item_idxs, top_scores), start=1):
        if idx in idx_to_row.index:
            r = idx_to_row.loc[idx].to_dict()
            r["item_idx"] = int(idx)
            r["rank"]     = rank
            r["score"]    = float(score)
            rows.append(r)

    candidates_df = pd.DataFrame(rows)

    # ── Step 5: Optional LLM rerank ─────────────────────────────────────
    if request.use_llm_rerank and not is_cold_start and len(candidates_df) > 0:
        pipeline_stages.append("llm_rerank")
        user_profile = {}
        if "user_profile_lookup" in artefacts:
            user_profile = artefacts["user_profile_lookup"].get(user_id, {})

        from api.llm_reranker import llm_rerank
        candidates_df = llm_rerank(user_profile, candidates_df)
        # Re-assign rank after reranking
        candidates_df["rank"] = range(1, len(candidates_df) + 1)

    # ── Step 6: Build response ───────────────────────────────────────────
    recommendations = []
    for _, row in candidates_df.iterrows():
        recommendations.append({
            "item_idx":      int(row.get("item_idx", 0)),
            "item_id":       str(row.get("item_id", "")),
            "grade":         str(row.get("grade", "")),
            "purpose":       str(row.get("purpose", "")),
            "term":          str(row.get("term", "")),
            "int_rate":      float(row.get("int_rate", 0.0)),
            "loan_amnt":     float(row.get("loan_amnt", 0.0)),
            "positive_rate": float(row.get("positive_rate", 0.0)),
            "rank":          int(row.get("rank", 0)),
            "score":         float(row.get("score", 0.0)),
        })

    return {
        "user_id":         user_id,
        "n_returned":      len(recommendations),
        "pipeline_stages": pipeline_stages,
        "recommendations": recommendations,
    }
