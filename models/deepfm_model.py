"""
models/deepfm_model.py
-----------------------
DeepFM model in PyTorch.

Reference: Guo et al. (2017) — "DeepFM: A Factorization-Machine based Neural
Network for CTR Prediction."

Architecture:
  Input = concatenation of sparse (ID) embeddings + dense (continuous) features

  FM Layer  — captures 2nd-order feature interactions via embedding inner products
  Deep Layer — MLP tower on the same embedding + dense features
  Output    — FM_output + Deep_output → Sigmoid
"""

import torch
import torch.nn as nn
from typing import List, Optional


class FMLayer(nn.Module):
    """
    Factorisation Machine 2nd-order interaction term.
    Input: stacked field embeddings  (B, n_fields, emb_dim)
    Output: scalar interaction score (B,)
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # sum_square - square_sum trick for efficiency
        sum_sq  = embeddings.sum(dim=1) ** 2          # (B, emb_dim)
        sq_sum  = (embeddings ** 2).sum(dim=1)         # (B, emb_dim)
        return 0.5 * (sum_sq - sq_sum).sum(dim=-1)    # (B,)


class DeepFM(nn.Module):
    """
    Parameters
    ----------
    sparse_field_dims : list[int]
        Cardinality of each sparse (categorical) field.
        For this project: [n_users, n_items, n_grades, n_purposes, n_terms, …]
    dense_dim : int
        Number of continuous input features (user + item scalars).
    emb_dim : int
        Embedding dimension for each sparse field.
    mlp_layers : list[int]
        Hidden layer sizes of the deep component.
    dropout : float
    """

    def __init__(
        self,
        sparse_field_dims: List[int],
        dense_dim: int = 0,
        emb_dim: int = 16,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        self.emb_dim    = emb_dim
        self.dense_dim  = dense_dim
        self.n_fields   = len(sparse_field_dims)

        # Separate embedding table per sparse field
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim)
            for dim in sparse_field_dims
        ])

        # FM layer (stateless; just the formula)
        self.fm = FMLayer()

        # FM bias terms (1st-order)
        self.bias_layers = nn.ModuleList([
            nn.Embedding(dim, 1)
            for dim in sparse_field_dims
        ])
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # Deep tower
        deep_input_dim = self.n_fields * emb_dim + dense_dim
        layers = []
        in_dim = deep_input_dim
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(mlp_layers[-1], 1)

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in self.bias_layers:
            nn.init.zeros_(emb.weight)
        for m in self.deep.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        sparse_inputs: torch.Tensor,          # (B, n_fields)  integer indices
        dense_inputs: Optional[torch.Tensor] = None,  # (B, dense_dim)
    ) -> torch.Tensor:                        # (B,) scores in (0,1)

        # ── Embeddings ──────────────────────────────────────────────────
        field_embs = torch.stack([
            self.embeddings[i](sparse_inputs[:, i])
            for i in range(self.n_fields)
        ], dim=1)  # (B, n_fields, emb_dim)

        # ── FM component ────────────────────────────────────────────────
        fm_first_order = torch.stack([
            self.bias_layers[i](sparse_inputs[:, i]).squeeze(-1)
            for i in range(self.n_fields)
        ], dim=1).sum(dim=1)  # (B,)
        fm_second_order = self.fm(field_embs)  # (B,)
        fm_out = self.fm_bias + fm_first_order + fm_second_order  # (B,)

        # ── Deep component ───────────────────────────────────────────────
        flat_emb = field_embs.flatten(start_dim=1)  # (B, n_fields * emb_dim)
        if dense_inputs is not None and self.dense_dim > 0:
            deep_in = torch.cat([flat_emb, dense_inputs], dim=-1)
        else:
            deep_in = flat_emb

        deep_out = self.deep_output(self.deep(deep_in)).squeeze(-1)  # (B,)

        # ── Output ──────────────────────────────────────────────────────
        return self.sigmoid(fm_out + deep_out)  # (B,)


# ── Convenience factory ─────────────────────────────────────────────────────

def build_deepfm(n_users: int, n_items: int,
                 dense_dim: int = 0) -> DeepFM:
    """
    Default DeepFM configuration for LendingClub.
    Sparse fields: user_id, item_id  (2 fields — matches training/inference input)
    """
    sparse_field_dims = [n_users, n_items]
    return DeepFM(
        sparse_field_dims=sparse_field_dims,
        dense_dim=dense_dim,
        emb_dim=16,
        mlp_layers=[256, 128, 64],
        dropout=0.2,
    )
