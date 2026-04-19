"""
ranking/repay_predictor.py
---------------------------
Inference wrapper for the XGBoost repayment-probability predictor.

Loads the saved XGBoost model and pre-loaded feature arrays.
Exposes a single method:
  predict(user_idxs, item_idxs) → float32 array of repayment probabilities

Used by:
  - ranking/dataset.py  (offline: augments training batches)
  - ranking/predictor.py (online: augments NeuMF input at inference)
"""

import os
import numpy as np
import xgboost as xgb

SAVED_DIR  = os.path.join("models", "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "xgboost_repay.json")


class RepayPredictor:
    """
    Parameters
    ----------
    model_path    : str         Path to saved XGBoost model (.json)
    user_features : np.ndarray  Shape (n_users, F_u)
    item_features : np.ndarray  Shape (n_items, F_i)
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        user_features: np.ndarray = None,
        item_features: np.ndarray = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}. "
                "Run ranking/train_xgboost.py first."
            )
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.user_features = user_features
        self.item_features = item_features
        print(f"[RepayPredictor] Loaded XGBoost model from {model_path}")

        calibrator_path = os.path.join(SAVED_DIR, "repay_calibrator.pkl")
        self.calibrator = None
        if os.path.exists(calibrator_path):
            import pickle
            with open(calibrator_path, "rb") as f:
                self.calibrator = pickle.load(f)
            print(f"[RepayPredictor] Loaded Isotonic Calibrator from {calibrator_path}")
        else:
            print("[RepayPredictor] No calibrator found. Using raw XGBoost probabilities.")

    def predict_from_features(
        self,
        user_feat_vec: np.ndarray,
        item_idxs: np.ndarray,
    ) -> np.ndarray:
        """
        Predict repayment probability using a raw user feature vector.
        Used for cold-start users who provide demographics in the request.

        Parameters
        ----------
        user_feat_vec : 1-D float32 array of shape (F_u,)
        item_idxs     : 1-D int array of shape (N,)

        Returns
        -------
        probs : float32 array of shape (N,)
        """
        n = len(item_idxs)
        user_feats = np.tile(user_feat_vec, (n, 1))   # (N, F_u)
        X = np.hstack([user_feats, self.item_features[item_idxs]])
        dmat = xgb.DMatrix(X)
        raw_probs = self.model.predict(dmat).astype(np.float32)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs).astype(np.float32)
        return raw_probs

    def predict(
        self,
        user_idxs: np.ndarray,
        item_idxs: np.ndarray,
    ) -> np.ndarray:
        """
        Predict repayment probability for a batch of (user, item) pairs.

        Parameters
        ----------
        user_idxs : 1-D int array  (length N)
        item_idxs : 1-D int array  (length N)

        Returns
        -------
        probs : float32 array of shape (N,) with values in [0, 1]
        """
        X = np.hstack([
            self.user_features[user_idxs],
            self.item_features[item_idxs],
        ])
        dmat = xgb.DMatrix(X)
        raw_probs = self.model.predict(dmat).astype(np.float32)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs).astype(np.float32)
        return raw_probs
