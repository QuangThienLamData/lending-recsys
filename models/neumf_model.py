"""
models/neumf_model.py
----------------------
Neural Matrix Factorization (NeuMF) model in PyTorch.

Reference: He et al. (2017) — "Neural Collaborative Filtering."

Architecture:
  Two branches, fused at the output:

  1. GMF branch (Generalised Matrix Factorisation)
     user_emb_gmf ⊙ item_emb_gmf  →  element-wise product

  2. MLP branch
     concat(user_emb_mlp, item_emb_mlp, user_features, item_features)
     → FC layers → hidden representation

  Output:
     concat(gmf_out, mlp_out) → Linear(→1) → Sigmoid → score ∈ (0,1)
"""

import torch
import torch.nn as nn
from typing import List


class NeuMF(nn.Module):
    """
    Parameters
    ----------
    n_users : int
    n_items : int
    emb_dim : int           Size of each ID embedding (default 32)
    user_feat_dim : int     Dimension of pre-computed user feature vector (0 = skip)
    item_feat_dim : int     Dimension of pre-computed item feature vector (0 = skip)
    mlp_layers : list[int]  Hidden layer sizes for the MLP branch
    dropout : float         Dropout probability in MLP layers
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        user_feat_dim: int = 0,
        item_feat_dim: int = 0,
        mlp_layers: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.emb_dim       = emb_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim

        # ── GMF embeddings ──────────────────────────────────────────────
        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(n_items, emb_dim)

        # ── MLP embeddings ──────────────────────────────────────────────
        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.item_emb_mlp = nn.Embedding(n_items, emb_dim)

        # ── MLP tower ───────────────────────────────────────────────────
        mlp_input_dim = 2 * emb_dim + user_feat_dim + item_feat_dim
        mlp_modules   = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            mlp_modules += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # ── Output layer ────────────────────────────────────────────────
        # GMF output dim = emb_dim (element-wise product)
        # MLP output dim = mlp_layers[-1]
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)
        self.sigmoid       = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_idx: torch.Tensor,          # (B,)
        item_idx: torch.Tensor,          # (B,)
        user_feats: torch.Tensor = None, # (B, user_feat_dim)  optional
        item_feats: torch.Tensor = None, # (B, item_feat_dim)  optional
    ) -> torch.Tensor:                   # (B,) scores in (0,1)

        # GMF branch
        ug = self.user_emb_gmf(user_idx)   # (B, d)
        ig = self.item_emb_gmf(item_idx)   # (B, d)
        gmf_out = ug * ig                  # (B, d)

        # MLP branch
        um = self.user_emb_mlp(user_idx)   # (B, d)
        im = self.item_emb_mlp(item_idx)   # (B, d)
        mlp_input = torch.cat([um, im], dim=-1)  # (B, 2d)

        if user_feats is not None and self.user_feat_dim > 0:
            mlp_input = torch.cat([mlp_input, user_feats], dim=-1)
        if item_feats is not None and self.item_feat_dim > 0:
            mlp_input = torch.cat([mlp_input, item_feats], dim=-1)

        mlp_out = self.mlp(mlp_input)      # (B, mlp_layers[-1])

        # Fusion
        fused  = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.output_layer(fused).squeeze(-1)  # (B,)
        return self.sigmoid(logits)


# ── Convenience factory ─────────────────────────────────────────────────────

def build_neumf(n_users: int, n_items: int,
                user_feat_dim: int = 0, item_feat_dim: int = 0) -> NeuMF:
    """Default NeuMF configuration used throughout the project."""
    return NeuMF(
        n_users=n_users,
        n_items=n_items,
        emb_dim=32,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
    )
