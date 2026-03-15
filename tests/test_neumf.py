"""
tests/test_neumf.py
--------------------
Smoke tests for models/neumf_model.py and models/deepfm_model.py.

Run with:  pytest tests/test_neumf.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.neumf_model import NeuMF, build_neumf
from models.deepfm_model import DeepFM, build_deepfm


N_USERS = 100
N_ITEMS = 50
BATCH   = 16


class TestNeuMF:
    def _model(self, user_feat=0, item_feat=0):
        return build_neumf(N_USERS, N_ITEMS,
                           user_feat_dim=user_feat, item_feat_dim=item_feat)

    def test_forward_no_features(self):
        model = self._model()
        u = torch.randint(0, N_USERS, (BATCH,))
        i = torch.randint(0, N_ITEMS, (BATCH,))
        out = model(u, i)
        assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Sigmoid output must be in [0,1]"

    def test_forward_with_features(self):
        U_DIM, I_DIM = 8, 6
        model = self._model(user_feat=U_DIM, item_feat=I_DIM)
        u     = torch.randint(0, N_USERS, (BATCH,))
        i     = torch.randint(0, N_ITEMS, (BATCH,))
        uf    = torch.randn(BATCH, U_DIM)
        itf   = torch.randn(BATCH, I_DIM)
        out   = model(u, i, uf, itf)
        assert out.shape == (BATCH,)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_output_range(self):
        model = self._model()
        u     = torch.randint(0, N_USERS, (1000,))
        i     = torch.randint(0, N_ITEMS, (1000,))
        with torch.no_grad():
            out = model(u, i)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_grad_flows(self):
        model = self._model()
        u     = torch.randint(0, N_USERS, (BATCH,))
        i     = torch.randint(0, N_ITEMS, (BATCH,))
        loss  = model(u, i).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self):
        model = self._model()
        model.eval()
        u = torch.randint(0, N_USERS, (BATCH,))
        i = torch.randint(0, N_ITEMS, (BATCH,))
        with torch.no_grad():
            out1 = model(u, i)
            out2 = model(u, i)
        assert torch.allclose(out1, out2), "Eval mode must be deterministic"


class TestDeepFM:
    def _model(self, dense_dim=0):
        return build_deepfm(N_USERS, N_ITEMS, dense_dim=dense_dim)

    def test_forward_no_dense(self):
        model = self._model()
        sparse_in = torch.stack([
            torch.randint(0, N_USERS, (BATCH,)),
            torch.randint(0, N_ITEMS, (BATCH,)),
            torch.randint(0, 7,       (BATCH,)),  # grade
            torch.randint(0, 15,      (BATCH,)),  # purpose
            torch.randint(0, 2,       (BATCH,)),  # term
        ], dim=1)
        out = model(sparse_in)
        assert out.shape == (BATCH,)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_forward_with_dense(self):
        DENSE = 12
        model = self._model(dense_dim=DENSE)
        sparse_in = torch.stack([
            torch.randint(0, N_USERS, (BATCH,)),
            torch.randint(0, N_ITEMS, (BATCH,)),
            torch.randint(0, 7,       (BATCH,)),
            torch.randint(0, 15,      (BATCH,)),
            torch.randint(0, 2,       (BATCH,)),
        ], dim=1)
        dense_in = torch.randn(BATCH, DENSE)
        out = model(sparse_in, dense_in)
        assert out.shape == (BATCH,)

    def test_grad_flows(self):
        model = self._model()
        sparse_in = torch.stack([
            torch.randint(0, N_USERS, (BATCH,)),
            torch.randint(0, N_ITEMS, (BATCH,)),
            torch.zeros(BATCH, dtype=torch.long),
            torch.zeros(BATCH, dtype=torch.long),
            torch.zeros(BATCH, dtype=torch.long),
        ], dim=1)
        loss = model(sparse_in).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
