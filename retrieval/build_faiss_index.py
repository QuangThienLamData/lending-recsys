"""
retrieval/build_faiss_index.py
-------------------------------
Loads ALS item embeddings, L2-normalises them (so inner product = cosine
similarity), and builds a FAISS index.

Index strategy:
  n_items < 50_000  → IndexFlatIP   (exact, brute-force inner product)
  n_items >= 50_000 → IndexIVFFlat  (approximate; n_cells = sqrt(n_items))

Output:
  models/saved/faiss.index

Usage:
  python -m retrieval.build_faiss_index [--index-type flat|ivf]
"""

import os
import argparse
import numpy as np

SAVED_DIR      = os.path.join("models", "saved")
ITEM_EMB_PATH  = os.path.join(SAVED_DIR, "als_item_embeddings.npy")
FAISS_IDX_PATH = os.path.join(SAVED_DIR, "faiss.index")

IVF_N_CELLS_MIN  = 64
IVF_N_CELLS_MAX  = 4096
IVF_THRESHOLD    = 50_000   # use IVF when n_items exceeds this


def build_flat(item_emb: np.ndarray):
    import faiss
    d     = item_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_emb)
    return index


def build_ivf(item_emb: np.ndarray):
    import faiss
    d       = item_emb.shape[1]
    n_items = item_emb.shape[0]
    n_cells = int(np.clip(np.sqrt(n_items), IVF_N_CELLS_MIN, IVF_N_CELLS_MAX))
    print(f"  IVF: n_cells = {n_cells}")

    quantiser = faiss.IndexFlatIP(d)
    index     = faiss.IndexIVFFlat(quantiser, d, n_cells,
                                   faiss.METRIC_INNER_PRODUCT)
    index.train(item_emb)
    index.add(item_emb)
    index.nprobe = max(1, n_cells // 8)   # search 1/8 of cells at query time
    return index


def main(index_type: str = "auto"):
    os.makedirs(SAVED_DIR, exist_ok=True)

    print(f"[build_faiss_index] Loading item embeddings from {ITEM_EMB_PATH} …")
    item_emb = np.load(ITEM_EMB_PATH).astype("float32")
    n_items, d = item_emb.shape
    print(f"  Shape: {item_emb.shape}")

    # L2-normalise so inner product equals cosine similarity
    import faiss
    faiss.normalize_L2(item_emb)
    print("  L2-normalised embeddings")

    # Choose index type
    if index_type == "auto":
        index_type = "ivf" if n_items >= IVF_THRESHOLD else "flat"
    print(f"  Index type: {index_type}")

    if index_type == "flat":
        index = build_flat(item_emb)
    else:
        index = build_ivf(item_emb)

    faiss.write_index(index, FAISS_IDX_PATH)
    print(f"[build_faiss_index] Saved index → {FAISS_IDX_PATH}")
    print(f"  Index total vectors: {index.ntotal}  d={d}")
    print("[build_faiss_index] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from item embeddings")
    parser.add_argument("--index-type", choices=["flat", "ivf", "auto"],
                        default="auto",
                        help="flat=exact search, ivf=approximate, auto=decide by size")
    args = parser.parse_args()
    main(index_type=args.index_type)
