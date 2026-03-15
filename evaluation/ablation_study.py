"""
evaluation/ablation_study.py
-----------------------------
Compares three pipeline configurations side-by-side on the test split:

  Config 1 — Retrieval only     : ALS + FAISS, no ranking model
  Config 2 — Retrieval + Ranking: ALS + FAISS + NeuMF/DeepFM
  Config 3 — Full pipeline      : ALS + FAISS + NeuMF/DeepFM + LLM rerank
             (Config 3 requires a valid LLM API key in the environment)

Outputs a Markdown summary table to stdout and saves a CSV to:
  evaluation/ablation_results.csv

Usage:
  python -m evaluation.ablation_study [--k 10] [--n-users 500]
"""

import os
import json
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import faiss
import pandas as pd

SAVED_DIR     = os.path.join("models", "saved")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_CSV       = os.path.join("evaluation", "ablation_results.csv")


def load_artefacts():
    from ranking.predictor import RankingPredictor
    print("[ablation] Loading artefacts …")
    with open(os.path.join(SAVED_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    user_emb    = np.load(os.path.join(SAVED_DIR, "als_user_embeddings.npy"))
    faiss_index = faiss.read_index(os.path.join(SAVED_DIR, "faiss.index"))
    predictor   = RankingPredictor()
    test_mat    = sp.load_npz(os.path.join(PROCESSED_DIR, "test_interactions.npz"))
    item_lookup = pd.read_csv(os.path.join(PROCESSED_DIR, "item_lookup.csv"))
    return enc, user_emb, faiss_index, predictor, test_mat, item_lookup


def retrieval_only(user_emb, faiss_index, test_mat, users, k, pool):
    """Config 1: rank candidates by ALS dot-product score (no reranker)."""
    from evaluation.metrics import compute_metrics
    all_rec, all_gt = [], []

    for u in users:
        gt = set(test_mat.getrow(u).nonzero()[1])
        if not gt:
            continue
        u_vec = user_emb[u].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        D, I = faiss_index.search(u_vec, pool)
        # Sort by FAISS score (inner product — higher = better)
        scores   = D[0]
        top_k    = I[0][np.argsort(scores)[::-1][:k]].tolist()
        all_rec.append(top_k)
        all_gt.append(gt)

    return compute_metrics(all_rec, all_gt, k=k), len(all_rec)


def retrieval_plus_ranking(user_emb, faiss_index, predictor, test_mat, users, k, pool):
    """Config 2: FAISS retrieval → NeuMF/DeepFM ranking."""
    from evaluation.metrics import compute_metrics
    all_rec, all_gt = [], []

    for u in users:
        gt = set(test_mat.getrow(u).nonzero()[1])
        if not gt:
            continue
        u_vec = user_emb[u].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I      = faiss_index.search(u_vec, pool)
        candidates = I[0]
        scores     = predictor.score_candidates(int(u), candidates)
        top_k      = candidates[np.argsort(scores)[::-1][:k]].tolist()
        all_rec.append(top_k)
        all_gt.append(gt)

    return compute_metrics(all_rec, all_gt, k=k), len(all_rec)


def retrieval_plus_ranking_plus_llm(
    user_emb, faiss_index, predictor, test_mat,
    enc, item_lookup, users, k, pool
):
    """Config 3: FAISS → NeuMF → LLM rerank."""
    from evaluation.metrics import compute_metrics
    from api.llm_reranker import llm_rerank

    provider = os.getenv("LLM_PROVIDER", "openai")
    has_key  = (
        (provider == "openai"    and "OPENAI_API_KEY"    in os.environ) or
        (provider == "anthropic" and "ANTHROPIC_API_KEY" in os.environ)
    )
    if not has_key:
        print(f"  [ablation] No LLM API key for provider '{provider}'. "
              "Skipping Config 3.")
        return None, 0

    all_rec, all_gt = [], []
    idx_to_row = item_lookup.set_index("item_idx")
    user_classes = enc["user_enc"].classes_

    for u in users:
        gt = set(test_mat.getrow(u).nonzero()[1])
        if not gt:
            continue
        u_vec = user_emb[u].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I       = faiss_index.search(u_vec, pool)
        candidates = I[0]
        scores     = predictor.score_candidates(int(u), candidates)
        top_k_idxs = candidates[np.argsort(scores)[::-1][:k]]

        # Build candidates DataFrame for LLM
        rows = []
        for idx in top_k_idxs:
            if idx in idx_to_row.index:
                r = idx_to_row.loc[idx].to_dict()
                r["item_idx"] = int(idx)
                rows.append(r)
        cand_df = pd.DataFrame(rows)

        # LLM rerank with minimal user profile
        reranked = llm_rerank(user_profile={}, candidates=cand_df)
        top_k    = reranked["item_idx"].tolist()[:k]

        all_rec.append(top_k)
        all_gt.append(gt)

    return compute_metrics(all_rec, all_gt, k=k), len(all_rec)


def run_ablation(k: int = 10, n_users: int = 500, pool: int = 100):
    enc_obj, user_emb, faiss_index, predictor, test_mat, item_lookup = load_artefacts()

    with open(os.path.join(SAVED_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)

    rows, _ = test_mat.nonzero()
    all_users = np.unique(rows)
    if len(all_users) > n_users:
        rng   = np.random.default_rng(42)
        users = rng.choice(all_users, n_users, replace=False)
    else:
        users = all_users

    print(f"[ablation] Running on {len(users)} test users  k={k}  pool={pool}\n")

    # ── Config 1: Retrieval only ─────────────────────────────────────────
    print("Config 1: Retrieval only (ALS + FAISS) …")
    m1, n1 = retrieval_only(user_emb, faiss_index, test_mat, users, k, pool)

    # ── Config 2: Retrieval + Ranking ────────────────────────────────────
    print("Config 2: Retrieval + Ranking (ALS + FAISS + NeuMF) …")
    m2, n2 = retrieval_plus_ranking(user_emb, faiss_index, predictor,
                                     test_mat, users, k, pool)

    # ── Config 3: Full pipeline ──────────────────────────────────────────
    print("Config 3: Full pipeline (+ LLM rerank) …")
    m3, n3 = retrieval_plus_ranking_plus_llm(
        user_emb, faiss_index, predictor, test_mat,
        enc, item_lookup, users, k, pool
    )

    # ── Summary table ────────────────────────────────────────────────────
    configs = ["Retrieval only", "Retrieval + Ranking"]
    metrics_list = [m1, m2]
    if m3 is not None:
        configs.append("Retrieval + Ranking + LLM")
        metrics_list.append(m3)

    metric_names = list(m1.keys())
    df = pd.DataFrame(
        {cfg: [m[mn] for mn in metric_names] for cfg, m in zip(configs, metrics_list)},
        index=metric_names,
    )

    print("\n" + "═" * 70)
    print("ABLATION STUDY RESULTS")
    print("═" * 70)
    print(df.to_markdown(floatfmt=".4f"))
    print("═" * 70 + "\n")

    # Save CSV
    os.makedirs("evaluation", exist_ok=True)
    df.to_csv(OUT_CSV)
    print(f"[ablation] Saved results → {OUT_CSV}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--pool",    type=int, default=100)
    args = parser.parse_args()
    run_ablation(k=args.k, n_users=args.n_users, pool=args.pool)
