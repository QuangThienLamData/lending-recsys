"""
evaluation/evaluate_pipeline.py
---------------------------------
Evaluates the full recommendation pipeline on the test split.

For each user in the test set:
  1. Retrieve candidates via FAISS (ALS user embedding)
  2. Score with ranking model
  3. Compare top-K against the user's test-set interactions

Reports Recall@K, NDCG@K, HitRate@K, MRR to stdout and saves results
to evaluation/results.json.

Usage:
  python -m evaluation.evaluate_pipeline [--k 10] [--n-users 2000]
"""

import os
import json
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import faiss

SAVED_DIR     = os.path.join("models", "saved")
PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_PATH  = os.path.join("evaluation", "results.json")


def load_artefacts():
    import torch
    from ranking.predictor import RankingPredictor

    print("[evaluate] Loading artefacts …")
    with open(os.path.join(SAVED_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)

    user_emb    = np.load(os.path.join(SAVED_DIR, "als_user_embeddings.npy"))
    faiss_index = faiss.read_index(os.path.join(SAVED_DIR, "faiss.index"))
    predictor   = RankingPredictor()
    test_mat    = sp.load_npz(os.path.join(PROCESSED_DIR, "test_interactions.npz"))

    print(f"  test_mat: {test_mat.shape}  nnz={test_mat.nnz:,}")
    return enc, user_emb, faiss_index, predictor, test_mat


def evaluate(k: int = 10, n_users: int = 2000, retrieval_pool: int = 100):
    from evaluation.metrics import compute_metrics

    enc, user_emb, faiss_index, predictor, test_mat = load_artefacts()
    n_items = test_mat.shape[1]

    # Users who have at least one positive test interaction
    rows, _ = test_mat.nonzero()
    candidate_users = np.unique(rows)
    if len(candidate_users) > n_users:
        rng = np.random.default_rng(42)
        candidate_users = rng.choice(candidate_users, n_users, replace=False)

    print(f"[evaluate] Evaluating {len(candidate_users)} users  k={k}  pool={retrieval_pool}")

    all_recommended  = []
    all_ground_truth = []

    for u in candidate_users:
        gt = set(test_mat.getrow(u).nonzero()[1])
        if not gt:
            continue

        # FAISS retrieval
        u_vec = user_emb[u].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I = faiss_index.search(u_vec, retrieval_pool)
        candidates = I[0]

        # Ranking model
        scores      = predictor.score_candidates(int(u), candidates)
        sorted_idx  = np.argsort(scores)[::-1][:k]
        top_k       = candidates[sorted_idx].tolist()

        all_recommended.append(top_k)
        all_ground_truth.append(gt)

    metrics = compute_metrics(all_recommended, all_ground_truth, k=k)

    print("\n── Evaluation Results ──────────────────────────────")
    for name, val in metrics.items():
        print(f"  {name:15s}: {val:.4f}")
    print("────────────────────────────────────────────────────\n")

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"k": k, "n_users": len(all_recommended), "metrics": metrics}, f, indent=2)
    print(f"[evaluate] Results saved → {RESULTS_PATH}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation pipeline on test set")
    parser.add_argument("--k",        type=int, default=10,   help="Recall/NDCG cut-off")
    parser.add_argument("--n-users",  type=int, default=2000, help="Max users to evaluate")
    parser.add_argument("--pool",     type=int, default=100,  help="FAISS retrieval pool size")
    args = parser.parse_args()
    evaluate(k=args.k, n_users=args.n_users, retrieval_pool=args.pool)
