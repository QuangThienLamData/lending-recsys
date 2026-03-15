"""
models/als_model.py
--------------------
Pure PyTorch implementation of Alternating Least Squares (ALS) for implicit
collaborative filtering.

Reference: Hu, Koren & Volinsky (2008) — "Collaborative Filtering for
Implicit Feedback Datasets."

Closed-form update per factor dimension:
  U_u = (V^T C_u V + λI)^{-1} V^T C_u p_u
  V_i = (U^T C_i U + λI)^{-1} U^T C_i p_i

where:
  p_ui = 1 if interaction exists else 0  (preference)
  c_ui = 1 + α * r_ui                    (confidence; r_ui = raw count)
  α = confidence scaling factor (default 40)
"""

import torch
import numpy as np
import scipy.sparse as sp
from typing import Optional


class ALSModel:
    """
    Alternating Least Squares matrix factorisation for implicit feedback.

    Parameters
    ----------
    n_users : int
    n_items : int
    n_factors : int      Embedding dimensionality (default 64)
    n_iter : int         Number of ALS iterations (default 20)
    reg : float          L2 regularisation λ (default 0.1)
    alpha : float        Confidence scaling α (default 40)
    device : str         'cpu' or 'cuda'
    seed : int
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 64,
        n_iter: int = 20,
        reg: float = 0.1,
        alpha: float = 40.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_users   = n_users
        self.n_items   = n_items
        self.n_factors = n_factors
        self.n_iter    = n_iter
        self.reg       = reg
        self.alpha     = alpha
        self.device    = torch.device(device)

        torch.manual_seed(seed)
        # Initialise factors with small random values
        self.U = torch.randn(n_users, n_factors, device=self.device) * 0.01
        self.V = torch.randn(n_items, n_factors, device=self.device) * 0.01

    # ------------------------------------------------------------------ #
    # Core ALS solver
    # ------------------------------------------------------------------ #

    def _als_step_users(self, R_csr: sp.csr_matrix):
        """Update all user factors given fixed item factors V."""
        V = self.V  # (n_items, d)
        VtV = V.T @ V  # (d, d)
        reg_eye = self.reg * torch.eye(self.n_factors, device=self.device)

        for u in range(self.n_users):
            # Items interacted by user u
            row_start = R_csr.indptr[u]
            row_end   = R_csr.indptr[u + 1]
            item_ids  = R_csr.indices[row_start:row_end]
            counts    = R_csr.data[row_start:row_end]

            if len(item_ids) == 0:
                continue  # no interactions; keep random init

            # Confidence weights c_ui = 1 + alpha * r_ui
            c = torch.tensor(1.0 + self.alpha * counts, dtype=torch.float32,
                             device=self.device)
            # Relevant item embeddings
            V_u = V[item_ids]            # (n_i, d)
            # A = V^T C_u V + λI
            A = VtV + (V_u * (c - 1).unsqueeze(1)).T @ V_u + reg_eye
            # b = V^T C_u p_u   (p_u = all-ones for observed items)
            b = (V_u * c.unsqueeze(1)).sum(dim=0)  # (d,)

            self.U[u] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def _als_step_items(self, R_csc: sp.csc_matrix):
        """Update all item factors given fixed user factors U."""
        U = self.U  # (n_users, d)
        UtU = U.T @ U  # (d, d)
        reg_eye = self.reg * torch.eye(self.n_factors, device=self.device)

        for i in range(self.n_items):
            col_start = R_csc.indptr[i]
            col_end   = R_csc.indptr[i + 1]
            user_ids  = R_csc.indices[col_start:col_end]
            counts    = R_csc.data[col_start:col_end]

            if len(user_ids) == 0:
                continue

            c   = torch.tensor(1.0 + self.alpha * counts, dtype=torch.float32,
                               device=self.device)
            U_i = U[user_ids]  # (n_u, d)
            A   = UtU + (U_i * (c - 1).unsqueeze(1)).T @ U_i + reg_eye
            b   = (U_i * c.unsqueeze(1)).sum(dim=0)

            self.V[i] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    def fit(self, R: sp.csr_matrix, verbose: bool = True):
        """
        Train ALS on the implicit interaction matrix R.

        Parameters
        ----------
        R : scipy.sparse.csr_matrix
            User × item interaction matrix (values = raw counts or 1s).
        """
        R_csr = R.astype(np.float32)
        R_csc = R_csr.tocsc()

        for it in range(1, self.n_iter + 1):
            self._als_step_users(R_csr)
            self._als_step_items(R_csc)

            if verbose:
                # Approximate training loss (sum of squared errors on observed)
                rows, cols = R_csr.nonzero()
                sample = min(50_000, len(rows))
                idx = np.random.choice(len(rows), sample, replace=False)
                u_s = torch.tensor(rows[idx], device=self.device)
                i_s = torch.tensor(cols[idx], device=self.device)
                preds = (self.U[u_s] * self.V[i_s]).sum(dim=1)
                mse   = ((preds - 1.0) ** 2).mean().item()
                print(f"  Iter {it:>2}/{self.n_iter}  approx MSE (positives): {mse:.4f}")

        print("[ALSModel] Training complete.")

    # ------------------------------------------------------------------ #
    # Accessor
    # ------------------------------------------------------------------ #

    @property
    def user_factors(self) -> np.ndarray:
        return self.U.cpu().numpy()

    @property
    def item_factors(self) -> np.ndarray:
        return self.V.cpu().numpy()

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: Optional[np.ndarray] = None) -> np.ndarray:
        """Return top-K item indices for a single user by dot-product score."""
        u_vec = self.U[user_idx]   # (d,)
        scores = (self.V @ u_vec).cpu().numpy()  # (n_items,)
        if exclude_seen is not None:
            scores[exclude_seen] = -np.inf
        return np.argsort(scores)[::-1][:top_k]
