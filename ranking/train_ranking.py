"""
ranking/train_ranking.py
-------------------------
Training loop for the NeuMF / DeepFM ranking model.

Features:
  - BCELoss binary cross-entropy
  - Adam optimiser with LR scheduler
  - Early stopping on validation NDCG@10
  - Saves best model checkpoint to models/saved/ranking_model.pt
  - Logs training and validation metrics to stdout

Usage:
  python -m ranking.train_ranking [--model neumf|deepfm] [--epochs 20] [--batch 2048]
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SAVED_DIR      = os.path.join("models", "saved")
PROCESSED_DIR  = os.path.join("data", "processed")
MODEL_PATH     = os.path.join(SAVED_DIR, "ranking_model.pt")
META_PATH      = os.path.join(PROCESSED_DIR, "feature_meta.json")


# ── Evaluation helpers ──────────────────────────────────────────────────────

def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    import math
    dcg  = sum(1.0 / math.log2(r + 2)
               for r, item in enumerate(recommended[:k]) if item in ground_truth)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate(model, val_mat: sp.csr_matrix,
             user_features: np.ndarray, item_features: np.ndarray,
             device: torch.device, k: int = 10,
             n_eval_users: int = 1000,
             model_type: str = "neumf") -> float:
    """Compute mean NDCG@K over a sample of users from the val matrix."""
    model.eval()
    rows, _ = val_mat.nonzero()
    users_with_pos = np.unique(rows)
    if len(users_with_pos) > n_eval_users:
        users_with_pos = np.random.choice(users_with_pos, n_eval_users, replace=False)

    n_items = val_mat.shape[1]
    all_item_idx = torch.arange(n_items, device=device)
    ndcg_scores  = []

    for u in users_with_pos:
        gt_items = set(val_mat.getrow(u).nonzero()[1])
        u_tensor = torch.tensor([u] * n_items, dtype=torch.long, device=device)
        uf = (torch.tensor(user_features[[u] * n_items], dtype=torch.float32, device=device)
              if user_features is not None else None)
        itf = (torch.tensor(item_features, dtype=torch.float32, device=device)
               if item_features is not None else None)

        if model_type == "neumf":
            scores = model(u_tensor, all_item_idx, uf, itf).cpu().numpy()
        else:
            sparse_in = torch.stack([u_tensor, all_item_idx], dim=1)
            dense_in  = None
            if uf is not None and itf is not None:
                dense_in = torch.cat([uf, itf], dim=-1)
            elif uf is not None:
                dense_in = uf
            elif itf is not None:
                dense_in = itf
            scores = model(sparse_in, dense_in).cpu().numpy()

        top_k   = np.argsort(scores)[::-1][:k].tolist()
        ndcg_scores.append(ndcg_at_k(top_k, gt_items, k))

    return float(np.mean(ndcg_scores))


# ── Training loop ───────────────────────────────────────────────────────────

def train(
    model_type: str = "neumf",
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    early_stop_patience: int = 5,
    device_str: str = "auto",
):
    os.makedirs(SAVED_DIR, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train_ranking] Device: {device}")

    # Load feature metadata
    user_feat_dim = item_feat_dim = 0
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        user_feat_dim = meta.get("user_feat_dim", 0)
        item_feat_dim = meta.get("item_feat_dim", 0)
        n_users       = meta["n_users"]
        n_items       = meta["n_items"]
    else:
        mat = sp.load_npz(os.path.join(PROCESSED_DIR, "train_interactions.npz"))
        n_users, n_items = mat.shape

    print(f"  n_users={n_users:,}  n_items={n_items}  "
          f"user_feat_dim={user_feat_dim}  item_feat_dim={item_feat_dim}")

    # ── Datasets & loaders ──────────────────────────────────────────────
    from ranking.dataset import RankingDataset

    train_ds = RankingDataset("train", use_features=(user_feat_dim > 0))
    val_mat  = sp.load_npz(os.path.join(PROCESSED_DIR, "val_interactions.npz"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Feature arrays for eval ─────────────────────────────────────────
    uf_path = os.path.join(PROCESSED_DIR, "user_features.npy")
    if_path = os.path.join(PROCESSED_DIR, "item_features.npy")
    user_features = np.load(uf_path) if os.path.exists(uf_path) and user_feat_dim > 0 else None
    item_features = np.load(if_path) if os.path.exists(if_path) and item_feat_dim > 0 else None

    # ── Model ───────────────────────────────────────────────────────────
    if model_type == "neumf":
        from models.neumf_model import build_neumf
        model = build_neumf(n_users, n_items, user_feat_dim, item_feat_dim)
    else:
        from models.deepfm_model import build_deepfm
        model = build_deepfm(n_users, n_items, dense_dim=user_feat_dim + item_feat_dim)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_type}  trainable params: {total_params:,}")

    # ── Optimiser & Loss ────────────────────────────────────────────────
    criterion  = nn.BCELoss()
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=2, verbose=True)

    best_ndcg   = 0.0
    patience_ctr = 0

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            user_idx  = batch["user_idx"].to(device)
            item_idx  = batch["item_idx"].to(device)
            labels    = batch["label"].to(device)
            uf = batch.get("user_feats")
            itf = batch.get("item_feats")
            if uf is not None:  uf  = uf.to(device)
            if itf is not None: itf = itf.to(device)

            if model_type == "neumf":
                preds = model(user_idx, item_idx, uf, itf)
            else:
                # DeepFM: build sparse_inputs tensor [user_idx, item_idx]
                sparse_in  = torch.stack([user_idx, item_idx], dim=1)
                dense_in   = None
                if uf is not None and itf is not None:
                    dense_in = torch.cat([uf, itf], dim=-1)
                elif uf is not None:
                    dense_in = uf
                elif itf is not None:
                    dense_in = itf
                preds = model(sparse_in, dense_in)

            loss = criterion(preds, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(train_ds)

        # Validation NDCG@10
        val_ndcg = evaluate(model, val_mat, user_features, item_features,
                            device, k=10, model_type=model_type)
        scheduler.step(val_ndcg)

        print(f"  Epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  "
              f"val_NDCG@10={val_ndcg:.4f}")

        # Save best
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (NDCG@10={best_ndcg:.4f}) → {MODEL_PATH}")
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print(f"  Early stopping triggered (no improvement for "
                      f"{early_stop_patience} epochs)")
                break

    print(f"\n[train_ranking] Training complete.  Best val NDCG@10 = {best_ndcg:.4f}")

    # Persist model type in metadata so the API can load the right class
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["model_type"] = model_type
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return best_ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuMF / DeepFM ranking model")
    parser.add_argument("--model",   choices=["neumf", "deepfm"], default="neumf")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch",   type=int,   default=2048)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--patience",type=int,   default=5)
    parser.add_argument("--device",  default="auto")
    args = parser.parse_args()
    train(model_type=args.model, epochs=args.epochs, batch_size=args.batch,
          lr=args.lr, early_stop_patience=args.patience, device_str=args.device)
