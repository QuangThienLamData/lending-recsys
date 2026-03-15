"""
retrieval/train_als.py
-----------------------
Orchestrates ALS training on the implicit interaction matrix and exports
user and item embedding arrays to models/saved/.

Usage:
  python -m retrieval.train_als [--factors 64] [--iterations 20] [--reg 0.1]
  python retrieval/train_als.py

Two backends are supported (selected via --backend flag):
  'implicit'  — uses the C++/CUDA optimised `implicit` library (fast, recommended)
  'pytorch'   — uses our pure-PyTorch ALSModel (educational, slower on large data)
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp

PROCESSED_DIR = os.path.join("data", "processed")
SAVED_DIR     = os.path.join("models", "saved")
USER_EMB_PATH = os.path.join(SAVED_DIR, "als_user_embeddings.npy")
ITEM_EMB_PATH = os.path.join(SAVED_DIR, "als_item_embeddings.npy")


def train_with_implicit(train_mat: sp.csr_matrix, factors: int,
                        iterations: int, reg: float) -> tuple:
    """Train ALS using the `implicit` library (recommended)."""
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        raise ImportError(
            "`implicit` library not found. "
            "Install it with: pip install implicit  "
            "or switch to --backend pytorch"
        )

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=reg,
        use_gpu=False,
        random_state=42,
    )
    # implicit expects item × user format
    # After fitting, model.user_factors → row factors (items here),
    # model.item_factors → column factors (users here). Swap to return (users, items).
    model.fit(train_mat.T.astype(np.float32), show_progress=True)
    return model.item_factors, model.user_factors


def train_with_pytorch(train_mat: sp.csr_matrix, factors: int,
                       iterations: int, reg: float) -> tuple:
    """Train ALS using our pure-PyTorch ALSModel."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.als_model import ALSModel

    n_users, n_items = train_mat.shape
    model = ALSModel(
        n_users=n_users,
        n_items=n_items,
        n_factors=factors,
        n_iter=iterations,
        reg=reg,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    model.fit(train_mat, verbose=True)
    return model.user_factors, model.item_factors


def main(factors: int = 64, iterations: int = 20,
         reg: float = 0.1, backend: str = "implicit"):
    os.makedirs(SAVED_DIR, exist_ok=True)

    print(f"[train_als] Loading train interaction matrix from {PROCESSED_DIR} …")
    train_mat = sp.load_npz(os.path.join(PROCESSED_DIR, "train_interactions.npz"))
    print(f"  Matrix shape: {train_mat.shape}  nnz: {train_mat.nnz:,}")

    print(f"[train_als] Training ALS (backend={backend}, factors={factors}, "
          f"iter={iterations}, reg={reg}) …")

    if backend == "implicit":
        user_emb, item_emb = train_with_implicit(train_mat, factors, iterations, reg)
    else:
        user_emb, item_emb = train_with_pytorch(train_mat, factors, iterations, reg)

    print(f"  user_emb shape: {user_emb.shape}")
    print(f"  item_emb shape: {item_emb.shape}")

    np.save(USER_EMB_PATH, user_emb.astype(np.float32))
    np.save(ITEM_EMB_PATH, item_emb.astype(np.float32))
    print(f"[train_als] Saved embeddings →")
    print(f"  {USER_EMB_PATH}")
    print(f"  {ITEM_EMB_PATH}")
    print("[train_als] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALS and export embeddings")
    parser.add_argument("--factors",    type=int,   default=64)
    parser.add_argument("--iterations", type=int,   default=20)
    parser.add_argument("--reg",        type=float, default=0.1)
    parser.add_argument("--backend",    choices=["implicit", "pytorch"],
                        default="implicit",
                        help="'implicit' (fast C++ library) or 'pytorch' (educational)")
    args = parser.parse_args()
    main(factors=args.factors, iterations=args.iterations,
         reg=args.reg, backend=args.backend)
