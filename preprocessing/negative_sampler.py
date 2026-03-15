"""
preprocessing/negative_sampler.py
-----------------------------------
Generates (user_idx, neg_item_idx) pairs for ranking model training
using popularity-weighted sampling.

Strategy:
  - For each positive (user, item) interaction, sample k_neg items the
    user has NOT interacted with.
  - Item sampling probability ∝ item popularity^0.75 (à la word2vec).
    This creates harder negatives (popular items the user ignored).

Outputs written to data/processed/:
  neg_samples_train.npy  — int32 array (N, 2) cols: [user_idx, neg_item_idx]
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

OUT_DIR = os.path.join("data", "processed")
K_NEG   = 4   # negative samples per positive interaction
SEED    = 42


def popularity_weights(train_mat: sp.csr_matrix, power: float = 0.75) -> np.ndarray:
    """Item sampling probability proportional to popularity^power."""
    item_counts = np.asarray(train_mat.sum(axis=0)).flatten()
    weights = item_counts ** power
    weights = weights.astype(np.float64)
    weights /= weights.sum()   # normalize after cast to avoid float32→float64 precision drift
    return weights


def sample_negatives(
    train_mat: sp.csr_matrix,
    k_neg: int = K_NEG,
    seed: int = SEED,
) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    n_items = train_mat.shape[1]
    weights = popularity_weights(train_mat)

    # Convert to lil for fast row access
    train_lil = train_mat.tolil()

    records = []
    users, items = train_mat.nonzero()
    positive_pairs = list(zip(users, items))

    print(f"[negative_sampler] Sampling {k_neg} negatives for {len(positive_pairs):,} positives …")
    for user_idx, _ in tqdm(positive_pairs, desc="Negative sampling"):
        seen = set(train_lil.rows[user_idx])
        neg_count = 0
        attempts  = 0
        max_attempts = k_neg * 20

        while neg_count < k_neg and attempts < max_attempts:
            candidate = rng.choice(n_items, p=weights)
            if candidate not in seen:
                records.append((user_idx, candidate))
                seen.add(candidate)
                neg_count += 1
            attempts += 1

    neg_array = np.array(records, dtype=np.int32)
    print(f"  Generated {len(neg_array):,} negative samples")
    return neg_array


def main(k_neg: int = K_NEG):
    train_mat = sp.load_npz(os.path.join(OUT_DIR, "train_interactions.npz"))
    neg_samples = sample_negatives(train_mat, k_neg=k_neg)
    out_path = os.path.join(OUT_DIR, "neg_samples_train.npy")
    np.save(out_path, neg_samples)
    print(f"[negative_sampler] Saved {out_path}  shape={neg_samples.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-neg", type=int, default=K_NEG,
                        help="Number of negatives per positive")
    args = parser.parse_args()
    main(k_neg=args.k_neg)
