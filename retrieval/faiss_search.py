"""
retrieval/faiss_search.py
--------------------------
Standalone FAISS retrieval module for the online inference pipeline.

Responsibility: given a user_idx, return the top-N candidate item indices
using approximate nearest-neighbour (ANN) search on ALS embeddings.

This module is kept intentionally thin — it knows nothing about the ranking
model, item metadata, or API schemas.  It is called by api/recommender.py
after user encoding and before the DeepFM scoring step.

Usage (standalone smoke-test):
    python -m retrieval.faiss_search --user-idx 0 --top-n 20
"""

import os
import logging
import numpy as np

import faiss

logger = logging.getLogger(__name__)

# Default paths — override via environment variables in production
_SAVED_DIR       = os.getenv("MODEL_DIR",        os.path.join("models", "saved"))
_FAISS_IDX_PATH  = os.getenv("FAISS_INDEX_PATH", os.path.join(_SAVED_DIR, "faiss.index"))
_USER_EMB_PATH   = os.path.join(_SAVED_DIR, "als_user_embeddings.npy")
_ITEM_EMB_PATH   = os.path.join(_SAVED_DIR, "als_item_embeddings.npy")


class FAISSRetriever:
    """
    Wraps a pre-built FAISS index and ALS user embeddings for fast
    approximate nearest-neighbour candidate retrieval.

    Load once at startup (expensive), call .retrieve() many times (cheap).

    Parameters
    ----------
    index_path   : str   Path to the serialised FAISS index (.index file).
    user_emb_path: str   Path to the ALS user embedding matrix (.npy, float32).
    item_emb_path: str   Path to the ALS item embedding matrix (.npy, float32).
                         Only used to validate dimensionality; not queried at
                         runtime because the FAISS index already stores item vecs.
    nprobe       : int   For IVF indexes: number of Voronoi cells to scan at
                         query time.  Higher = more accurate but slower.
                         Ignored for flat (exact) indexes.

    Attributes
    ----------
    index         : faiss.Index     The loaded FAISS index.
    user_emb      : np.ndarray      Shape (n_users, d), float32.
    n_users       : int
    embedding_dim : int
    """

    def __init__(
        self,
        index_path:    str = _FAISS_IDX_PATH,
        user_emb_path: str = _USER_EMB_PATH,
        item_emb_path: str = _ITEM_EMB_PATH,
        nprobe:        int = 8,
    ):
        # ── Load FAISS index ──────────────────────────────────────────────────
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run retrieval/build_faiss_index.py first."
            )
        self.index = faiss.read_index(index_path)

        # If the index is an IVF variant, tune nprobe for the accuracy/speed
        # trade-off.  For flat (brute-force) indexes this attribute doesn't
        # exist and the assignment is silently ignored.
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
            logger.debug("FAISSRetriever: set nprobe=%d", nprobe)

        logger.info(
            "FAISSRetriever: loaded FAISS index from '%s'  "
            "(ntotal=%d, d=%d)",
            index_path, self.index.ntotal, self.index.d,
        )

        # ── Load user embeddings ──────────────────────────────────────────────
        if not os.path.exists(user_emb_path):
            raise FileNotFoundError(
                f"User embeddings not found at '{user_emb_path}'. "
                "Run retrieval/train_als.py first."
            )
        self.user_emb = np.load(user_emb_path).astype("float32")
        self.n_users, self.embedding_dim = self.user_emb.shape

        logger.info(
            "FAISSRetriever: loaded user_emb %s from '%s'",
            self.user_emb.shape, user_emb_path,
        )

        # ── Sanity check: embedding dims must match ───────────────────────────
        if self.embedding_dim != self.index.d:
            raise ValueError(
                f"Dimension mismatch: user_emb has d={self.embedding_dim} "
                f"but FAISS index has d={self.index.d}. "
                "Rebuild the index with the same ALS embeddings."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_idx: int,
        top_n:    int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top-N candidate item indices for a single user.

        Steps:
          1. Look up the user's ALS embedding vector.
          2. L2-normalise it so inner-product search equals cosine similarity.
             (The FAISS index stores L2-normalised item embeddings, built by
             retrieval/build_faiss_index.py.)
          3. Run FAISS index.search() to retrieve the nearest neighbours.

        Parameters
        ----------
        user_idx : int   Row index into user_emb (0-based, post label-encoding).
        top_n    : int   Number of candidates to return.

        Returns
        -------
        candidate_idxs : np.ndarray  shape (top_n,) int64  — item indices
        distances      : np.ndarray  shape (top_n,) float32 — FAISS scores
                         (inner-product / cosine similarity, higher = more similar)

        Raises
        ------
        IndexError  if user_idx is out of range for the loaded user_emb matrix.
        """
        if user_idx < 0 or user_idx >= self.n_users:
            raise IndexError(
                f"user_idx={user_idx} is out of range "
                f"[0, {self.n_users - 1}] for the loaded embedding matrix."
            )

        # Step 1: look up user vector
        # Shape: (1, d) — FAISS expects a 2-D float32 query matrix
        user_vec = self.user_emb[user_idx].reshape(1, -1).copy()

        # Step 2: L2-normalise so inner-product == cosine similarity
        # faiss.normalize_L2 operates in-place on the array.
        faiss.normalize_L2(user_vec)

        # Step 3: ANN search
        # D — distances (inner products), shape (1, top_n)
        # I — indices,   shape (1, top_n)
        D, I = self.index.search(user_vec, top_n)

        # Squeeze the batch dimension: (1, top_n) → (top_n,)
        candidate_idxs = I[0].astype(np.int64)
        distances      = D[0].astype(np.float32)

        # FAISS may return -1 for padding when the index has fewer than top_n
        # vectors.  Filter those out to avoid downstream index-out-of-bounds.
        valid          = candidate_idxs >= 0
        candidate_idxs = candidate_idxs[valid]
        distances      = distances[valid]

        logger.debug(
            "FAISSRetriever.retrieve: user_idx=%d  top_n=%d  returned=%d",
            user_idx, top_n, len(candidate_idxs),
        )
        return candidate_idxs, distances

    def retrieve_batch(
        self,
        user_idxs: list[int],
        top_n:     int = 50,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Batch retrieval for multiple users in a single FAISS call.
        More efficient than calling retrieve() in a Python loop when
        serving batch inference or offline evaluation.

        Parameters
        ----------
        user_idxs : list[int]   User indices to query.
        top_n     : int         Candidates per user.

        Returns
        -------
        List of (candidate_idxs, distances) tuples, one per user.
        """
        if not user_idxs:
            return []

        # Stack user vectors into a single (B, d) matrix
        vecs = self.user_emb[user_idxs].copy()   # (B, d)
        faiss.normalize_L2(vecs)

        D, I = self.index.search(vecs, top_n)     # (B, top_n) each

        results = []
        for idxs_row, dists_row in zip(I, D):
            valid = idxs_row >= 0
            results.append((
                idxs_row[valid].astype(np.int64),
                dists_row[valid].astype(np.float32),
            ))
        return results

    # ── Convenience repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FAISSRetriever("
            f"n_users={self.n_users}, "
            f"n_items={self.index.ntotal}, "
            f"d={self.embedding_dim})"
        )


# ── Standalone smoke-test ─────────────────────────────────────────────────────

def _smoke_test(user_idx: int = 0, top_n: int = 20):
    """Quick CLI test: load the retriever and print candidates for one user."""
    logging.basicConfig(level=logging.INFO)
    retriever = FAISSRetriever()
    print(repr(retriever))
    candidates, scores = retriever.retrieve(user_idx=user_idx, top_n=top_n)
    print(f"\nTop-{top_n} candidates for user_idx={user_idx}:")
    for rank, (idx, score) in enumerate(zip(candidates, scores), start=1):
        print(f"  [{rank:>3}]  item_idx={idx:>5}  score={score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test FAISSRetriever")
    parser.add_argument("--user-idx", type=int, default=0)
    parser.add_argument("--top-n",    type=int, default=20)
    args = parser.parse_args()
    _smoke_test(user_idx=args.user_idx, top_n=args.top_n)
