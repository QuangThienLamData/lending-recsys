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

APPROVAL_THRESHOLD = 0.65


def _build_approval_item_feats(request, artefacts: dict):
    """
    Build item feature vector from the user's specific loan request
    (loan_amnt, int_rate, purpose).  Grade and term are unknown so
    OneHotEncoder outputs zeros for those columns (handle_unknown='ignore').
    Returns float32 array of shape (item_feat_dim,), or None.
    """
    transformer = artefacts.get("item_transformer")
    if transformer is None or request.loan_amnt_request is None:
        return None

    row = {
        "int_rate":  request.int_rate_request  if request.int_rate_request  is not None else 12.0,
        "loan_amnt": request.loan_amnt_request,
        "grade":     "Unknown",
        "purpose":   request.purpose_request   if request.purpose_request   is not None else "other",
        "term":      request.term_request      if request.term_request      is not None else "Unknown",
    }
    return transformer.transform(pd.DataFrame([row])).astype(np.float32)[0]


def _compute_approval_score(user_feat_vec, approval_item_feats, predictor) -> float:
    """Run XGBoost with the user's specific loan request to get approval probability."""
    if (user_feat_vec is None or approval_item_feats is None
            or predictor.repay_predictor is None):
        return 0.0
    X = np.hstack([user_feat_vec.reshape(1, -1),
                   approval_item_feats.reshape(1, -1)])
    import xgboost as xgb
    prob = predictor.repay_predictor.model.predict(xgb.DMatrix(X))
    return float(prob[0])


def _build_cold_start_user_feats(request, artefacts: dict):
    """
    Build a user feature vector for cold-start users.
    Priority:
      1. Look up demographics from user_profile_lookup (by user_id)
      2. Fall back to manually provided fields on the request
    Returns a float32 array of shape (user_feat_dim,), or None if not possible.
    """
    transformer = artefacts.get("user_transformer")
    if transformer is None:
        return None

    # 1. Try profile lookup first
    profile_lookup = artefacts.get("user_profile_lookup")
    row = None
    if profile_lookup is not None and request.user_id in profile_lookup.index:
        p = profile_lookup.loc[request.user_id]
        row = {
            "annual_inc":      float(p["annual_inc"]),
            "dti":             float(p["dti"]),
            "fico_range_low":  float(p["fico_range_low"]),
            "fico_range_high": float(p["fico_range_high"]),
            "home_ownership":  str(p["home_ownership"]),
            "addr_state":      str(p["addr_state"]),
        }

    # 2. Fall back to request fields
    if row is None:
        if all(v is None for v in [
            request.annual_inc, request.dti,
            request.fico_range_low, request.fico_range_high,
        ]):
            return None
        top_states = artefacts.get("top_states", [])
        state = request.addr_state or "other"
        if top_states and state not in top_states:
            state = "other"
        row = {
            "annual_inc":      request.annual_inc,
            "dti":             request.dti,
            "fico_range_low":  request.fico_range_low,
            "fico_range_high": request.fico_range_high,
            "home_ownership":  request.home_ownership or "Unknown",
            "addr_state":      state,
        }

    top_states = artefacts.get("top_states", [])
    if top_states and row["addr_state"] not in top_states:
        row["addr_state"] = "other"

    df = pd.DataFrame([row])
    return transformer.transform(df).astype(np.float32)[0]   # shape (F_u,)


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

    has_repay = predictor.repay_predictor is not None
    pipeline_stages = (
        ["retrieval", "repay_prediction", "ranking"]
        if has_repay else ["retrieval", "ranking"]
    )
    top_k     = request.top_k
    pool      = request.retrieval_pool
    # When LLM reranking is on, pass a wider set so the LLM can surface
    # items that the ranking model ranked outside the final top_k
    ranking_k = max(top_k, request.llm_pool) if request.use_llm_rerank else top_k

    # ── Step 1: Encode user ──────────────────────────────────────────────
    user_id = request.user_id
    known_users = set(user_enc.classes_)
    is_cold_start = user_id not in known_users

    cold_user_feats = None
    warm_user_feats = None
    if is_cold_start:
        logger.info("Cold-start user '%s' — returning popular items.", user_id)
        candidate_idxs  = _get_popular_items(item_lookup, pool)
        user_vec        = None
        cold_user_feats = _build_cold_start_user_feats(request, artefacts)
        if cold_user_feats is not None:
            logger.info("  Cold-start demographics provided — will use XGBoost scoring.")
    else:
        user_idx = int(user_enc.transform([user_id])[0])
        user_vec = user_emb[user_idx].astype("float32").reshape(1, -1)
        faiss.normalize_L2(user_vec)
        if predictor.user_features is not None:
            warm_user_feats = predictor.user_features[user_idx]

        # ── Step 2: FAISS retrieval ──────────────────────────────────────
        _, I = faiss_index.search(user_vec, pool)
        candidate_idxs = I[0]

    # ── Step 2b: Filter candidates by repayment rate ≥ 65% ──────────────
    # Run the XGBoost repay predictor (with calibrator) on all retrieved candidates
    # and keep only those whose estimated repayment probability meets the threshold.
    # For warm-start users we have user_idx; for cold-start we use cold_user_feats.
    # If neither is available, fall back to the historical positive_rate on the catalog.
    if predictor.repay_predictor is not None:
        if not is_cold_start:
            pre_repay = predictor.repay_predictor.predict(
                np.array([user_idx] * len(candidate_idxs)), candidate_idxs
            )
        elif cold_user_feats is not None:
            pre_repay = predictor.repay_predictor.predict_from_features(
                cold_user_feats, candidate_idxs
            )
        else:
            pre_repay = np.ones(len(candidate_idxs), dtype=np.float32)

        pass_mask      = pre_repay >= APPROVAL_THRESHOLD
        n_before       = len(candidate_idxs)
        candidate_idxs = candidate_idxs[pass_mask]
        logger.info(
            "  Retrieval filter: %d / %d candidates pass repay ≥ %.0f%%",
            len(candidate_idxs), n_before, APPROVAL_THRESHOLD * 100
        )
        if len(candidate_idxs) == 0:
            # Safety: if nothing passes, fall back to all candidates rather than
            # returning an empty recommendation list.
            logger.warning("  All candidates below threshold — skipping repay filter.")
            candidate_idxs = I[0] if not is_cold_start else _get_popular_items(item_lookup, pool)
    else:
        # No repay predictor loaded — filter using item catalog positive_rate
        item_pos_rate = item_lookup.set_index("item_idx")["positive_rate"]
        pass_mask = np.array([
            item_pos_rate.get(int(idx), 0.0) >= APPROVAL_THRESHOLD
            for idx in candidate_idxs
        ])
        candidate_idxs = candidate_idxs[pass_mask] if pass_mask.any() else candidate_idxs

    # ── Step 3: Ranking model ────────────────────────────────────────────
    if not is_cold_start:
        scores = predictor.score_candidates(user_idx, candidate_idxs)
        sorted_order  = np.argsort(scores)[::-1][:ranking_k]
        top_item_idxs = candidate_idxs[sorted_order]
        top_scores    = scores[sorted_order]
    elif cold_user_feats is not None and predictor.repay_predictor is not None:
        # Cold-start with demographics: rank popular items by XGBoost repay score
        xgb_scores   = predictor.repay_predictor.predict_from_features(
            cold_user_feats, candidate_idxs
        )
        sorted_order  = np.argsort(xgb_scores)[::-1][:ranking_k]
        top_item_idxs = candidate_idxs[sorted_order]
        top_scores    = xgb_scores[sorted_order]
    else:
        top_item_idxs = candidate_idxs[:ranking_k]
        top_scores    = np.zeros(len(top_item_idxs), dtype=np.float32)

    # ── Step 4: Build candidates DataFrame ──────────────────────────────
    # Compute XGBoost repay probability for each top item
    xgb_repay_probs = {}
    if not is_cold_start and predictor.repay_predictor is not None:
        repay_scores = predictor.repay_predictor.predict(
            np.array([user_idx] * len(top_item_idxs)), top_item_idxs
        )
        xgb_repay_probs = dict(zip(top_item_idxs.tolist(), repay_scores.tolist()))
    elif cold_user_feats is not None and predictor.repay_predictor is not None:
        # Already computed above — reuse top_scores as the repay probs
        xgb_repay_probs = dict(zip(top_item_idxs.tolist(), top_scores.tolist()))

    idx_to_row = item_lookup.set_index("item_idx")
    rows = []
    for rank, (idx, score) in enumerate(zip(top_item_idxs, top_scores), start=1):
        if idx in idx_to_row.index:
            r = idx_to_row.loc[idx].to_dict()
            r["item_idx"]       = int(idx)
            r["rank"]           = rank
            r["score"]          = float(score)
            r["xgb_repay_prob"] = float(xgb_repay_probs.get(int(idx), 0.0))
            rows.append(r)

    candidates_df = pd.DataFrame(rows)

    # (filter already applied at retrieval stage above)

    # ── Step 5: Optional LLM rerank ─────────────────────────────────────
    if request.use_llm_rerank and not is_cold_start and len(candidates_df) > 0:
        pipeline_stages.append("llm_rerank")

        # Inject items so the LLM has a robust candidate set.
        # If user_prompt is provided, we inject the entire catalog to let the LLM
        # freely pick the absolute best product across all purposes and grades.
        # Otherwise, just inject the requested purpose from the form.
        if getattr(request, "user_prompt", None) and getattr(request, "user_prompt").strip():
            purpose_idxs = item_lookup["item_idx"].values.astype(np.int64)
        elif request.purpose_request:
            purpose_idxs = item_lookup.loc[
                item_lookup["purpose"] == request.purpose_request, "item_idx"
            ].values.astype(np.int64)
        else:
            purpose_idxs = []
            
        already_in = set(candidates_df["item_idx"].tolist())
        new_idxs   = [i for i in purpose_idxs if i not in already_in]

        if new_idxs:
            new_idxs_arr = np.array(new_idxs, dtype=np.int64)
            # Score injected items with ranking model
            new_scores = predictor.score_candidates(user_idx, new_idxs_arr)
            # Build repay probs for injected items
            new_repay = {}
            if predictor.repay_predictor is not None:
                rp = predictor.repay_predictor.predict(
                    np.array([user_idx] * len(new_idxs_arr)), new_idxs_arr
                )
                new_repay = dict(zip(new_idxs_arr.tolist(), rp.tolist()))

            inject_rows = []
            for idx, score in zip(new_idxs_arr, new_scores):
                if idx in idx_to_row.index:
                    repay_prob = float(new_repay.get(int(idx), 0.0))
                    # Only inject items that pass the repay threshold
                    repay_check = repay_prob if repay_prob > 0 else float(
                        idx_to_row.loc[idx].get("positive_rate", 0.0)
                    )
                    if repay_check < APPROVAL_THRESHOLD:
                        continue
                    r = idx_to_row.loc[idx].to_dict()
                    r["item_idx"]       = int(idx)
                    r["rank"]           = 0   # placeholder
                    r["score"]          = float(score)
                    r["xgb_repay_prob"] = repay_prob
                    inject_rows.append(r)

            if inject_rows:
                candidates_df = pd.concat(
                    [candidates_df, pd.DataFrame(inject_rows)],
                    ignore_index=True,
                )

        user_profile = {}
        profile_lookup = artefacts.get("user_profile_lookup")
        if profile_lookup is not None and user_id in profile_lookup.index:
            user_profile = profile_lookup.loc[user_id].to_dict()

        from api.llm_reranker import llm_rerank
        loan_request = {
            "purpose":   request.purpose_request,
            "loan_amnt": request.loan_amnt_request,
            "int_rate":  request.int_rate_request,
        }
        candidates_df = llm_rerank(user_profile, candidates_df, loan_request, getattr(request, "user_prompt", None))
        # Trim to top_k after LLM reranking, then re-assign ranks
        candidates_df = candidates_df.head(top_k).reset_index(drop=True)
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
            "positive_rate":  float(row.get("positive_rate", 0.0)),
            "xgb_repay_prob": float(row.get("xgb_repay_prob", 0.0)),
            "rank":           int(row.get("rank", 0)),
            "score":          float(row.get("score", 0.0)),
        })

    # ── Approval score from user's specific loan request ────────────────
    approval_item_feats = _build_approval_item_feats(request, artefacts)
    user_feat_for_approval = warm_user_feats if not is_cold_start else cold_user_feats
    approval_score = _compute_approval_score(
        user_feat_for_approval, approval_item_feats, predictor
    )
    approved = approval_score >= APPROVAL_THRESHOLD

    # ── Step 7: SHAP explanation + improvement suggestions ───────────────
    shap_features = []
    improvements  = []
    raw_profile   = {}
    llm_advice    = ""

    if user_feat_for_approval is not None and approval_item_feats is not None:
        # Get the raw (unscaled) user profile for improvement suggestions
        profile_lookup = artefacts.get("user_profile_lookup")
        if profile_lookup is not None and user_id in profile_lookup.index:
            p = profile_lookup.loc[user_id]
            raw_profile = {
                "annual_inc":      float(p["annual_inc"]),
                "dti":             float(p["dti"]),
                "fico_range_low":  float(p["fico_range_low"]),
                "fico_range_high": float(p["fico_range_high"]),
                "home_ownership":  str(p["home_ownership"]),
                "addr_state":      str(p["addr_state"]),
            }
        elif cold_user_feats is not None:
            # Cold-start: use request fields
            raw_profile = {
                "annual_inc":      request.annual_inc,
                "dti":             request.dti,
                "fico_range_low":  request.fico_range_low,
                "fico_range_high": request.fico_range_high,
                "home_ownership":  request.home_ownership or "Unknown",
                "addr_state":      request.addr_state or "other",
            }

        if raw_profile:
            from api.explain import compute_shap_values, find_improvements, generate_llm_advice
            shap_features = compute_shap_values(
                user_feat_for_approval, approval_item_feats, predictor, artefacts
            )
            raw_loan_request = {
                "purpose":   request.purpose_request,
                "loan_amnt": request.loan_amnt_request,
                "term":      request.term_request,
                "int_rate":  request.int_rate_request,
            }
            improvements = find_improvements(
                raw_profile, raw_loan_request, approval_score, predictor, artefacts
            )
            llm_advice = generate_llm_advice(
                user_profile   = raw_profile,
                shap_features  = shap_features,
                improvements   = improvements,
                approval_score = approval_score,
                approved       = approved,
                loan_request   = {
                    "purpose":   request.purpose_request,
                    "loan_amnt": request.loan_amnt_request,
                    "term":      request.term_request,
                    "int_rate":  request.int_rate_request,
                },
            )

    return {
        "user_id":         user_id,
        "n_returned":      len(recommendations),
        "pipeline_stages": pipeline_stages,
        "recommendations": recommendations,
        "approved":        approved,
        "approval_score":  approval_score,
        "shap_features":   shap_features,
        "improvements":    improvements,
        "user_profile":    raw_profile,
        "llm_advice":      llm_advice,
    }
