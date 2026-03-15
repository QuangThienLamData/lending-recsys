"""
tests/test_api.py
------------------
Integration tests for the FastAPI application.

These tests mock the artefacts so no real trained models are needed.
Run with:  pytest tests/test_api.py -v

For a full end-to-end integration test against a running server use:
  LIVE=1 pytest tests/test_api.py -v
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers to build fake artefacts ──────────────────────────────────────────

N_USERS = 20
N_ITEMS = 10
DIM     = 8

def _make_fake_artefacts():
    """Build minimal in-memory artefacts that satisfy the recommender pipeline."""
    from sklearn.preprocessing import LabelEncoder

    user_ids = [str(i) for i in range(N_USERS)]
    item_ids = [f"A_purpose_{t}" for t in ["36 months", "60 months"]] * 5

    user_enc = LabelEncoder().fit(user_ids)
    item_enc = LabelEncoder().fit(item_ids)

    user_emb = np.random.randn(N_USERS, DIM).astype("float32")
    item_emb = np.random.randn(N_ITEMS, DIM).astype("float32")

    # Normalise for cosine search
    norms = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8
    item_emb_norm = item_emb / norms

    import faiss
    faiss.normalize_L2(user_emb)
    index = faiss.IndexFlatIP(DIM)
    index.add(item_emb_norm)

    # Fake ranking predictor (returns random scores)
    class FakePredictor:
        model_type = "neumf"

        def score_candidates(self, user_idx, candidate_item_idxs):
            return np.random.rand(len(candidate_item_idxs)).astype("float32")

    item_lookup = pd.DataFrame({
        "item_idx":      list(range(N_ITEMS)),
        "item_id":       item_ids[:N_ITEMS],
        "grade":         ["A"] * N_ITEMS,
        "purpose":       ["debt_consolidation"] * N_ITEMS,
        "term":          ["36 months"] * N_ITEMS,
        "int_rate":      [10.0] * N_ITEMS,
        "loan_amnt":     [10000.0] * N_ITEMS,
        "positive_rate": [0.85] * N_ITEMS,
        "n_loans":       [100] * N_ITEMS,
    })

    return {
        "user_enc":        user_enc,
        "item_enc":        item_enc,
        "user_emb":        user_emb,
        "item_emb":        item_emb_norm,
        "faiss_index":     index,
        "ranking_predictor": FakePredictor(),
        "item_lookup":     item_lookup,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with mocked artefacts injected at startup."""
    from fastapi.testclient import TestClient
    from api import main as main_module

    fake_arts = _make_fake_artefacts()
    main_module.artefacts.update(fake_arts)

    with TestClient(main_module.app) as c:
        yield c

    main_module.artefacts.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = resp = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_artefacts_listed(self, client):
        data = client.get("/health").json()
        assert len(data["artefacts_loaded"]) > 0


class TestRecommendEndpoint:
    def test_known_user(self, client):
        resp = client.post("/recommend", json={"user_id": "5", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "5"
        assert len(data["recommendations"]) <= 5
        assert "retrieval" in data["pipeline_stages"]
        assert "ranking"   in data["pipeline_stages"]

    def test_cold_start_user(self, client):
        resp = client.post("/recommend", json={"user_id": "UNKNOWN_USER", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) <= 3

    def test_top_k_respected(self, client):
        for k in [1, 3, 5]:
            resp = client.post("/recommend", json={"user_id": "2", "top_k": k})
            assert resp.status_code == 200
            assert len(resp.json()["recommendations"]) <= k

    def test_response_schema(self, client):
        resp = client.post("/recommend", json={"user_id": "0", "top_k": 2})
        data = resp.json()
        assert "user_id"         in data
        assert "n_returned"      in data
        assert "pipeline_stages" in data
        assert "recommendations" in data
        if data["recommendations"]:
            item = data["recommendations"][0]
            for field in ["item_idx", "item_id", "grade", "purpose",
                          "term", "int_rate", "loan_amnt", "positive_rate",
                          "rank", "score"]:
                assert field in item, f"Missing field: {field}"

    def test_top_k_validation(self, client):
        # top_k=0 should fail validation
        resp = client.post("/recommend", json={"user_id": "0", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_max_validation(self, client):
        # top_k=200 exceeds max of 100
        resp = client.post("/recommend", json={"user_id": "0", "top_k": 200})
        assert resp.status_code == 422


class TestItemsEndpoint:
    def test_items_200(self, client):
        resp = client.get("/items")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == N_ITEMS


class TestUserInfoEndpoint:
    def test_known_user_info(self, client):
        resp = client.get("/users/5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["known"] is True
        assert data["cold_start"] is False

    def test_unknown_user_info(self, client):
        resp = client.get("/users/GHOST_USER")
        assert resp.status_code == 200
        data = resp.json()
        assert data["known"] is False
        assert data["cold_start"] is True
