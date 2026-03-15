"""
tests/test_metrics.py
----------------------
Unit tests for evaluation/metrics.py.

Run with:  pytest tests/test_metrics.py -v
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    compute_metrics,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial_recall(self):
        # 1 out of 2 ground-truth items in top-3 → recall = 0.5
        assert recall_at_k([1, 4, 5], {1, 2}, k=3) == 0.5

    def test_empty_ground_truth(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_k_larger_than_list(self):
        # Only 2 items recommended; recall against {1, 2} should be 1.0
        assert recall_at_k([1, 2], {1, 2}, k=10) == 1.0


class TestPrecisionAtK:
    def test_full_precision(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k([4, 5], {1, 2}, k=2) == 0.0

    def test_half_precision(self):
        assert precision_at_k([1, 4], {1, 2}, k=2) == 0.5

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1, 2}, k=0) == 0.0


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k([1, 2], {1, 2}, k=2) == 1.0

    def test_zero_ndcg(self):
        assert ndcg_at_k([3, 4], {1, 2}, k=2) == 0.0

    def test_order_matters(self):
        # Relevant item at rank 2 vs rank 1 should yield lower NDCG
        ndcg_rank1 = ndcg_at_k([1, 2], {1}, k=2)
        ndcg_rank2 = ndcg_at_k([2, 1], {1}, k=2)
        assert ndcg_rank1 > ndcg_rank2

    def test_empty_ground_truth(self):
        assert ndcg_at_k([1, 2], set(), k=2) == 0.0

    def test_known_value(self):
        # Single hit at rank 1: DCG = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        assert abs(ndcg_at_k([1], {1}, k=1) - 1.0) < 1e-9

    def test_hit_at_rank_2(self):
        # Hit at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1.0
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        result   = ndcg_at_k([99, 1], {1}, k=2)
        assert abs(result - expected) < 1e-9


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0

    def test_no_hit(self):
        assert hit_rate_at_k([1, 2], {5}, k=2) == 0.0

    def test_hit_beyond_k(self):
        # Item 4 is relevant but appears at rank 4, beyond k=3
        assert hit_rate_at_k([1, 2, 3, 4], {4}, k=3) == 0.0


class TestMRR:
    def test_first_rank(self):
        assert mean_reciprocal_rank([1, 2, 3], {1}) == 1.0

    def test_second_rank(self):
        assert mean_reciprocal_rank([2, 1, 3], {1}) == 0.5

    def test_no_hit(self):
        assert mean_reciprocal_rank([1, 2, 3], {5}) == 0.0


class TestComputeMetrics:
    def test_batch_single_perfect(self):
        metrics = compute_metrics([[1, 2, 3]], [{1, 2, 3}], k=3)
        assert metrics["Recall@3"]    == 1.0
        assert metrics["NDCG@3"]      == 1.0
        assert metrics["HitRate@3"]   == 1.0

    def test_batch_multiple(self):
        rec = [[1, 2], [3, 4], [5, 6]]
        gt  = [{1}, {9}, {5}]
        m   = compute_metrics(rec, gt, k=2)
        # User 0: hit. User 1: miss. User 2: hit. → HitRate = 2/3
        assert abs(m["HitRate@2"] - 2 / 3) < 1e-9
