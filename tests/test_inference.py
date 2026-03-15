"""
tests/test_inference.py
------------------------
Unit tests for the online inference pipeline using unittest + unittest.mock.

Design philosophy
-----------------
These tests are intentionally FAST and ISOLATED.  They must pass in CI/CD
without any saved model weights, FAISS index files, or GPU access.

Every heavy dependency (FAISS index, PyTorch model, file I/O) is replaced
with a lightweight Mock object.  This lets us:

  1. Verify the ranking/predictor.py scoring logic in isolation.
  2. Verify the retrieval/faiss_search.py retrieval logic in isolation.
  3. Verify that the full pipeline (recommender.py) correctly sorts
     candidates and returns the highest-scoring items first.
  4. Verify cold-start behaviour without a real user encoder.

Run with:
    python -m pytest tests/test_inference.py -v
    # or using the stdlib runner directly:
    python -m unittest tests.test_inference -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# Make the project root importable regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# 1.  FAISSRetriever unit tests
# =============================================================================

class TestFAISSRetriever(unittest.TestCase):
    """
    Tests for retrieval/faiss_search.py :: FAISSRetriever.

    The real FAISS index and .npy files are mocked so no artefacts are needed.
    """

    N_USERS = 50
    N_ITEMS = 30
    DIM     = 16

    def _make_retriever(self, nprobe=4):
        """
        Build a FAISSRetriever whose __init__ file-loading is bypassed via
        patching.  Returns (retriever, mock_index).
        """
        import faiss
        from retrieval.faiss_search import FAISSRetriever

        # --- Build a real tiny FAISS index so search() works correctly -------
        # We use the real faiss library but with toy data — no file I/O.
        item_vecs = np.random.randn(self.N_ITEMS, self.DIM).astype("float32")
        faiss.normalize_L2(item_vecs)
        real_index = faiss.IndexFlatIP(self.DIM)
        real_index.add(item_vecs)

        # --- Fake user embeddings (no file needed) ----------------------------
        user_vecs = np.random.randn(self.N_USERS, self.DIM).astype("float32")

        # --- Patch os.path.exists so __init__ doesn't raise FileNotFoundError -
        with patch("retrieval.faiss_search.os.path.exists", return_value=True), \
             patch("retrieval.faiss_search.faiss.read_index", return_value=real_index), \
             patch("retrieval.faiss_search.np.load",          return_value=user_vecs):
            retriever = FAISSRetriever(nprobe=nprobe)

        return retriever, real_index

    # ── Construction tests ────────────────────────────────────────────────────

    def test_retriever_loads_correctly(self):
        """Attributes are set from the (mocked) index and user embeddings."""
        retriever, _ = self._make_retriever()
        self.assertEqual(retriever.n_users,       self.N_USERS)
        self.assertEqual(retriever.embedding_dim, self.DIM)
        self.assertEqual(retriever.index.ntotal,  self.N_ITEMS)

    def test_missing_index_raises_file_not_found(self):
        """FileNotFoundError when the FAISS index file doesn't exist."""
        from retrieval.faiss_search import FAISSRetriever
        with patch("retrieval.faiss_search.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                FAISSRetriever()

    # ── retrieve() tests ──────────────────────────────────────────────────────

    def test_retrieve_returns_correct_shapes(self):
        """retrieve() returns arrays of the requested length."""
        retriever, _ = self._make_retriever()
        top_n = 10
        idxs, dists = retriever.retrieve(user_idx=0, top_n=top_n)

        # At most top_n results (may be fewer if index is small)
        self.assertLessEqual(len(idxs),  top_n)
        self.assertLessEqual(len(dists), top_n)
        self.assertEqual(len(idxs), len(dists))

    def test_retrieve_returns_valid_item_indices(self):
        """All returned item indices are within [0, N_ITEMS)."""
        retriever, _ = self._make_retriever()
        idxs, _ = retriever.retrieve(user_idx=5, top_n=20)
        for idx in idxs:
            self.assertGreaterEqual(int(idx), 0)
            self.assertLess(int(idx), self.N_ITEMS)

    def test_retrieve_scores_are_descending(self):
        """
        FAISS returns neighbours in descending inner-product order.
        This verifies the retrieval pre-sorts candidates by ALS similarity.
        """
        retriever, _ = self._make_retriever()
        _, dists = retriever.retrieve(user_idx=3, top_n=15)
        self.assertEqual(
            list(dists),
            sorted(dists, reverse=True),
            "FAISS distances should be in descending order",
        )

    def test_retrieve_out_of_range_user_raises(self):
        """IndexError for a user_idx beyond the embedding matrix size."""
        retriever, _ = self._make_retriever()
        with self.assertRaises(IndexError):
            retriever.retrieve(user_idx=self.N_USERS + 999)

    def test_retrieve_negative_user_idx_raises(self):
        """IndexError for a negative user_idx."""
        retriever, _ = self._make_retriever()
        with self.assertRaises(IndexError):
            retriever.retrieve(user_idx=-1)

    # ── retrieve_batch() tests ────────────────────────────────────────────────

    def test_retrieve_batch_returns_one_result_per_user(self):
        """retrieve_batch() returns a list of length == len(user_idxs)."""
        retriever, _ = self._make_retriever()
        user_idxs = [0, 1, 2, 3]
        results = retriever.retrieve_batch(user_idxs, top_n=10)
        self.assertEqual(len(results), len(user_idxs))

    def test_retrieve_batch_empty_input(self):
        """retrieve_batch([]) returns an empty list, not an error."""
        retriever, _ = self._make_retriever()
        results = retriever.retrieve_batch([], top_n=10)
        self.assertEqual(results, [])

    # ── repr test ─────────────────────────────────────────────────────────────

    def test_repr_contains_key_info(self):
        retriever, _ = self._make_retriever()
        r = repr(retriever)
        self.assertIn("FAISSRetriever", r)
        self.assertIn(str(self.N_USERS), r)


# =============================================================================
# 2.  RankingPredictor (DeepFM) unit tests
# =============================================================================

class TestRankingPredictor(unittest.TestCase):
    """
    Tests for ranking/predictor.py :: RankingPredictor.

    The PyTorch model, state-dict loading, and feature files are all mocked.
    This means the tests run in milliseconds with no GPU or .pt file.
    """

    N_USERS = 30
    N_ITEMS = 20
    N_CANDS = 15   # candidates per scoring call

    def _make_predictor(self, model_type="deepfm"):
        """
        Instantiate RankingPredictor with every file-loading call mocked out.
        Returns (predictor, mock_model).
        """
        from ranking.predictor import RankingPredictor

        # --- The feature metadata that __init__ reads from disk ---------------
        fake_meta = {
            "n_users":       self.N_USERS,
            "n_items":       self.N_ITEMS,
            "user_feat_dim": 0,
            "item_feat_dim": 0,
            "model_type":    model_type,
        }

        # --- The torch model that predictor will call -------------------------
        # We make its forward() return a deterministic tensor so we can assert
        # on exact score values downstream.
        mock_torch_model = MagicMock()
        mock_torch_model.eval.return_value = mock_torch_model

        import torch
        # score_candidates calls model(u_tensor, i_tensor) or model(sparse_in)
        # We return ascending scores so the highest-index candidate wins.
        fake_scores = torch.linspace(0.1, 0.9, self.N_CANDS)
        mock_torch_model.return_value = fake_scores

        # --- Patch all I/O and model construction ----------------------------
        import builtins

        with patch("ranking.predictor.os.path.exists", return_value=True), \
             patch("ranking.predictor.json.load",       return_value=fake_meta), \
             patch("builtins.open",                     MagicMock()), \
             patch("ranking.predictor.torch.load",      return_value={}), \
             patch("ranking.predictor.build_deepfm",    return_value=mock_torch_model), \
             patch("ranking.predictor.build_neumf",     return_value=mock_torch_model):
            predictor = RankingPredictor(
                model_path="/fake/ranking_model.pt",
                meta_path="/fake/feature_meta.json",
                device="cpu",
            )

        # Expose the mock so tests can inspect calls
        predictor.model = mock_torch_model
        return predictor

    # ── score_candidates tests ────────────────────────────────────────────────

    def test_score_candidates_returns_correct_length(self):
        """score_candidates returns one score per candidate item."""
        predictor = self._make_predictor()
        candidates = np.arange(self.N_CANDS, dtype=np.int64)

        scores = predictor.score_candidates(user_idx=0, candidate_item_idxs=candidates)

        self.assertEqual(len(scores), self.N_CANDS)

    def test_score_candidates_returns_float32(self):
        """Scores must be float32 for downstream numpy operations."""
        predictor = self._make_predictor()
        candidates = np.arange(self.N_CANDS, dtype=np.int64)
        scores = predictor.score_candidates(user_idx=0, candidate_item_idxs=candidates)
        self.assertEqual(scores.dtype, np.float32)

    def test_score_candidates_are_probabilities(self):
        """DeepFM uses Sigmoid output, so all scores must be in [0, 1]."""
        predictor = self._make_predictor()
        candidates = np.arange(self.N_CANDS, dtype=np.int64)
        scores = predictor.score_candidates(user_idx=0, candidate_item_idxs=candidates)
        self.assertTrue(
            np.all(scores >= 0.0) and np.all(scores <= 1.0),
            f"Scores out of [0,1]: min={scores.min():.3f} max={scores.max():.3f}",
        )

    def test_pipeline_returns_highest_scoring_items_first(self):
        """
        Core ranking correctness test.

        Given a mock scorer that returns known scores (linspace 0.1→0.9),
        verify that argsort[::-1] selects the top-K highest-scoring candidates
        in the correct descending order — matching what recommender.py does.
        """
        predictor = self._make_predictor()
        candidates = np.arange(self.N_CANDS, dtype=np.int64)

        scores = predictor.score_candidates(user_idx=0, candidate_item_idxs=candidates)

        # Replicate the sort logic from api/recommender.py
        top_k         = 5
        sorted_order  = np.argsort(scores)[::-1][:top_k]
        top_item_idxs = candidates[sorted_order]
        top_scores    = scores[sorted_order]

        # The highest-scoring candidate should appear first
        self.assertEqual(
            top_item_idxs[0], candidates[-1],
            "The highest-scoring item (index N_CANDS-1) must be rank-1",
        )

        # Scores must be in non-increasing order
        for i in range(len(top_scores) - 1):
            self.assertGreaterEqual(
                top_scores[i], top_scores[i + 1],
                f"Score at rank {i+1} ({top_scores[i]:.4f}) is less than "
                f"score at rank {i+2} ({top_scores[i+1]:.4f})",
            )


# =============================================================================
# 3.  Full pipeline (run_recommendation_pipeline) unit tests
# =============================================================================

class TestRecommendationPipeline(unittest.TestCase):
    """
    Tests for api/recommender.py :: run_recommendation_pipeline.

    Both the FAISS index and the ranking predictor are replaced with Mocks
    so the test runs in CI without any saved artefacts.

    Scenario under test:
      - Known user 42 is passed in.
      - Fake FAISS returns items [3, 1, 2, 4, 0] with ascending dummy scores.
      - Fake predictor returns scores [0.1, 0.9, 0.5, 0.3, 0.7] for those items.
      - We assert that item 1 (score 0.9) is ranked first.
    """

    N_USERS = 100
    N_ITEMS = 20
    DIM     = 8

    def setUp(self):
        """Build fake artefacts once for all tests in this class."""
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        # ── Encoders ──────────────────────────────────────────────────────────
        user_ids = [str(i) for i in range(self.N_USERS)]
        self.user_enc = LabelEncoder().fit(user_ids)

        item_ids = [f"item_{i}" for i in range(self.N_ITEMS)]
        self.item_enc = LabelEncoder().fit(item_ids)

        # ── User embeddings (random; FAISS call is mocked) ────────────────────
        self.user_emb = np.random.randn(self.N_USERS, self.DIM).astype("float32")

        # ── Mock FAISS index ──────────────────────────────────────────────────
        # Define the exact candidates FAISS will "return"
        self.fake_candidates = np.array([3, 1, 2, 4, 0], dtype=np.int64)
        self.faiss_distances  = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

        mock_faiss = MagicMock()
        # index.search returns (D, I) each shaped (1, n)
        mock_faiss.search.return_value = (
            self.faiss_distances.reshape(1, -1),
            self.fake_candidates.reshape(1, -1),
        )
        self.mock_faiss = mock_faiss

        # ── Mock ranking predictor ─────────────────────────────────────────────
        # Scores indexed by position in self.fake_candidates
        # item at position 1 (item_idx=1) gets the highest score: 0.9
        self.fake_scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7], dtype=np.float32)

        mock_predictor = MagicMock()
        mock_predictor.score_candidates.return_value = self.fake_scores
        mock_predictor.model_type = "deepfm"
        self.mock_predictor = mock_predictor

        # ── Item lookup table (must contain all candidate indices) ─────────────
        self.item_lookup = pd.DataFrame({
            "item_idx":      list(range(self.N_ITEMS)),
            "item_id":       [f"A_debt_36m_{i}" for i in range(self.N_ITEMS)],
            "grade":         ["A"] * self.N_ITEMS,
            "purpose":       ["debt_consolidation"] * self.N_ITEMS,
            "term":          ["36 months"] * self.N_ITEMS,
            "int_rate":      [10.5] * self.N_ITEMS,
            "loan_amnt":     [12000.0] * self.N_ITEMS,
            "positive_rate": [0.88] * self.N_ITEMS,
            "n_loans":       [250] * self.N_ITEMS,
        })

        # ── Assemble the artefacts dict that recommender.py expects ───────────
        self.artefacts = {
            "user_enc":          self.user_enc,
            "item_enc":          self.item_enc,
            "user_emb":          self.user_emb,
            "faiss_index":       self.mock_faiss,
            "ranking_predictor": self.mock_predictor,
            "item_lookup":       self.item_lookup,
        }

    def _make_request(self, user_id: str = "42", top_k: int = 3,
                      use_llm_rerank: bool = False, retrieval_pool: int = 50):
        """Build a minimal RecommendRequest-like object."""
        req = MagicMock()
        req.user_id         = user_id
        req.top_k           = top_k
        req.use_llm_rerank  = use_llm_rerank
        req.retrieval_pool  = retrieval_pool
        return req

    # ── Known user tests ──────────────────────────────────────────────────────

    def test_known_user_triggers_faiss_search(self):
        """For a known user, the FAISS index must be queried exactly once."""
        from api.recommender import run_recommendation_pipeline

        run_recommendation_pipeline(self._make_request("42"), self.artefacts)

        self.mock_faiss.search.assert_called_once()

    def test_known_user_triggers_ranking(self):
        """For a known user, the predictor must score the FAISS candidates."""
        from api.recommender import run_recommendation_pipeline

        run_recommendation_pipeline(self._make_request("42"), self.artefacts)

        self.mock_predictor.score_candidates.assert_called_once()

    def test_top_item_has_highest_score(self):
        """
        Rank-1 item must be the one with score=0.9 (item_idx=1 in our setup).

        Fake scores: [0.1, 0.9, 0.5, 0.3, 0.7] for candidates [3, 1, 2, 4, 0]
        argsort[::-1] → position 1 (item_idx=1) wins with 0.9.
        """
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42", top_k=3), self.artefacts
        )

        recs = result["recommendations"]
        self.assertGreater(len(recs), 0, "No recommendations returned")

        top_item = recs[0]
        self.assertEqual(
            top_item["item_idx"], 1,
            f"Expected item_idx=1 (score=0.9) at rank 1, got {top_item['item_idx']}",
        )

    def test_scores_are_descending(self):
        """Returned recommendations must be sorted by score, highest first."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42", top_k=5), self.artefacts
        )
        scores = [r["score"] for r in result["recommendations"]]
        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
            f"Scores not in descending order: {scores}",
        )

    def test_top_k_is_respected(self):
        """Never return more recommendations than top_k."""
        from api.recommender import run_recommendation_pipeline

        for k in [1, 2, 3]:
            result = run_recommendation_pipeline(
                self._make_request("42", top_k=k), self.artefacts
            )
            self.assertLessEqual(
                len(result["recommendations"]), k,
                f"top_k={k} violated: got {len(result['recommendations'])} items",
            )

    def test_pipeline_stages_include_retrieval_and_ranking(self):
        """The response must declare both 'retrieval' and 'ranking' stages."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42"), self.artefacts
        )
        stages = result["pipeline_stages"]
        self.assertIn("retrieval", stages)
        self.assertIn("ranking",   stages)

    def test_response_fields_are_present(self):
        """Every item in the response must have the full expected schema."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42", top_k=2), self.artefacts
        )
        required_fields = [
            "item_idx", "item_id", "grade", "purpose",
            "term", "int_rate", "loan_amnt", "positive_rate", "rank", "score",
        ]
        for item in result["recommendations"]:
            for field in required_fields:
                self.assertIn(field, item, f"Missing field '{field}' in response item")

    def test_ranks_are_one_based_and_consecutive(self):
        """rank field must be 1, 2, 3, … without gaps."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42", top_k=3), self.artefacts
        )
        ranks = [r["rank"] for r in result["recommendations"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    # ── Cold-start user tests ─────────────────────────────────────────────────

    def test_cold_start_user_skips_faiss(self):
        """An unknown user must NOT trigger the FAISS search."""
        from api.recommender import run_recommendation_pipeline

        run_recommendation_pipeline(
            self._make_request("GHOST_9999999"), self.artefacts
        )
        # FAISS search must NOT have been called for a cold-start user
        self.mock_faiss.search.assert_not_called()

    def test_cold_start_user_skips_ranking(self):
        """An unknown user must NOT trigger the ranking model."""
        from api.recommender import run_recommendation_pipeline

        run_recommendation_pipeline(
            self._make_request("GHOST_9999999"), self.artefacts
        )
        self.mock_predictor.score_candidates.assert_not_called()

    def test_cold_start_user_returns_popular_items(self):
        """Cold-start returns items sorted by popularity, not by model score."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("GHOST_9999999", top_k=3), self.artefacts
        )
        # Must still return recommendations (popular items fallback)
        self.assertGreater(len(result["recommendations"]), 0)

    def test_cold_start_user_scores_are_zero(self):
        """Cold-start items have score=0.0 (no model was invoked)."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("GHOST_9999999", top_k=3), self.artefacts
        )
        for item in result["recommendations"]:
            self.assertEqual(
                item["score"], 0.0,
                f"Cold-start item score should be 0.0, got {item['score']}",
            )

    # ── Edge-case tests ───────────────────────────────────────────────────────

    def test_user_id_echoed_in_response(self):
        """The response must echo the exact user_id from the request."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42"), self.artefacts
        )
        self.assertEqual(result["user_id"], "42")

    def test_n_returned_matches_recommendations_length(self):
        """n_returned field must equal the actual length of recommendations."""
        from api.recommender import run_recommendation_pipeline

        result = run_recommendation_pipeline(
            self._make_request("42", top_k=3), self.artefacts
        )
        self.assertEqual(result["n_returned"], len(result["recommendations"]))


# =============================================================================
# 4.  Pydantic schema validation tests
# =============================================================================

class TestPydanticSchemas(unittest.TestCase):
    """
    Tests for api/schemas.py — validate that the Pydantic models enforce
    their constraints (field types, min/max bounds, required fields).
    """

    def test_recommend_request_defaults(self):
        """RecommendRequest should apply sensible defaults."""
        from api.schemas import RecommendRequest
        req = RecommendRequest(user_id="user_1")
        self.assertEqual(req.top_k,           10)
        self.assertEqual(req.use_llm_rerank,  False)
        self.assertEqual(req.retrieval_pool,  100)

    def test_recommend_request_top_k_too_low(self):
        """top_k=0 must raise a Pydantic ValidationError."""
        from api.schemas import RecommendRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            RecommendRequest(user_id="u", top_k=0)

    def test_recommend_request_top_k_too_high(self):
        """top_k=200 exceeds the max of 100 and must be rejected."""
        from api.schemas import RecommendRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            RecommendRequest(user_id="u", top_k=200)

    def test_recommend_request_valid_range(self):
        """top_k values at the boundary (1 and 100) must be accepted."""
        from api.schemas import RecommendRequest
        for k in [1, 100]:
            req = RecommendRequest(user_id="u", top_k=k)
            self.assertEqual(req.top_k, k)

    def test_health_response_default_status(self):
        """HealthResponse.status defaults to 'ok'."""
        from api.schemas import HealthResponse
        hr = HealthResponse()
        self.assertEqual(hr.status, "ok")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
