"""
evaluation/metrics.py
----------------------
Pure Python / NumPy implementations of the standard IR metrics used to
evaluate the recommendation pipeline.

All functions accept ranked item lists and ground-truth sets and are
independent of any ML framework.
"""

import math
import numpy as np
from typing import List, Set


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Recall@K: fraction of ground-truth items that appear in the top-K
    recommended items.

    Parameters
    ----------
    recommended  : ordered list of item indices (highest score first)
    ground_truth : set of relevant item indices for this user
    k            : cut-off rank

    Returns
    -------
    float in [0, 1]
    """
    if not ground_truth:
        return 0.0
    hits = len(set(recommended[:k]) & ground_truth)
    return hits / min(len(ground_truth), k)


def precision_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Precision@K: fraction of the top-K recommended items that are relevant.
    """
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & ground_truth)
    return hits / k


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Rewards relevant items appearing earlier in the ranked list.
    Uses binary relevance (1 if item in ground_truth, else 0).

    Parameters
    ----------
    recommended  : ordered list of item indices
    ground_truth : set of relevant item indices
    k            : cut-off rank

    Returns
    -------
    float in [0, 1]
    """
    if not ground_truth:
        return 0.0

    dcg  = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in ground_truth
    )
    # Ideal DCG: all top-min(|GT|,k) positions are hits
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """
    Hit Rate @ K: 1 if at least one ground-truth item appears in top-K, else 0.
    Useful as a per-user binary metric.
    """
    return float(bool(set(recommended[:k]) & ground_truth))


def mean_reciprocal_rank(recommended: List[int], ground_truth: Set[int]) -> float:
    """
    Mean Reciprocal Rank (MRR): reciprocal of the rank of the first relevant item.
    Returns 0 if no relevant item found.
    """
    for rank, item in enumerate(recommended, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


# ── Batch aggregation ────────────────────────────────────────────────────────

def compute_metrics(
    all_recommended: List[List[int]],
    all_ground_truth: List[Set[int]],
    k: int = 10,
) -> dict:
    """
    Compute mean Recall@K, Precision@K, NDCG@K, HitRate@K, MRR over all users.

    Parameters
    ----------
    all_recommended  : list[list[int]] — one ranked list per user
    all_ground_truth : list[set[int]]  — one ground-truth set per user
    k                : cut-off rank

    Returns
    -------
    dict with keys: recall, precision, ndcg, hit_rate, mrr
    """
    assert len(all_recommended) == len(all_ground_truth), \
        "Mismatch between number of recommended lists and ground-truth sets"

    recalls, precisions, ndcgs, hits, mrrs = [], [], [], [], []

    for rec, gt in zip(all_recommended, all_ground_truth):
        recalls.append(recall_at_k(rec, gt, k))
        precisions.append(precision_at_k(rec, gt, k))
        ndcgs.append(ndcg_at_k(rec, gt, k))
        hits.append(hit_rate_at_k(rec, gt, k))
        mrrs.append(mean_reciprocal_rank(rec, gt))

    return {
        f"Recall@{k}":    float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"NDCG@{k}":      float(np.mean(ndcgs)),
        f"HitRate@{k}":   float(np.mean(hits)),
        "MRR":            float(np.mean(mrrs)),
    }
