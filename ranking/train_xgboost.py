"""
ranking/train_xgboost.py
-------------------------
Train an XGBoost binary classifier to predict loan repayment probability.

Features : user_features[user_idx]  ||  item_features[item_idx]
Label    : 1 = repaid, 0 = defaulted  (same label used in build_interactions.py)

Outputs:
  models/saved/xgboost_repay.json   — trained XGBoost model
  Updates feature_meta.json         — sets repay_feat_dim = 1

Usage:
  python -m ranking.train_xgboost [--n-estimators 300] [--max-depth 6] [--lr 0.1]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

PROCESSED_DIR = os.path.join("data", "processed")
SAVED_DIR     = os.path.join("models", "saved")
META_PATH     = os.path.join(PROCESSED_DIR, "feature_meta.json")
MODEL_PATH    = os.path.join(SAVED_DIR, "xgboost_repay.json")

VAL_START  = "2017-01-01"
TEST_START = "2018-01-01"


def _build_X(user_idxs: np.ndarray, item_idxs: np.ndarray,
             user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
    return np.hstack([user_features[user_idxs], item_features[item_idxs]])


def train(n_estimators: int = 300, max_depth: int = 6, lr: float = 0.1):
    os.makedirs(SAVED_DIR, exist_ok=True)

    # ── Load feature arrays ──────────────────────────────────────────────────
    uf_path = os.path.join(PROCESSED_DIR, "user_features.npy")
    if_path = os.path.join(PROCESSED_DIR, "item_features.npy")
    if not os.path.exists(uf_path) or not os.path.exists(if_path):
        raise FileNotFoundError(
            "user_features.npy / item_features.npy not found. "
            "Run preprocessing/feature_engineering.py first."
        )
    user_features = np.load(uf_path)
    item_features = np.load(if_path)
    print(f"[train_xgboost] user_features {user_features.shape}  "
          f"item_features {item_features.shape}")

    # ── Load full labeled interaction DataFrame ──────────────────────────────
    interactions_path = os.path.join(PROCESSED_DIR, "interactions_all.parquet")
    df = pd.read_parquet(interactions_path)
    df["issue_d"] = pd.to_datetime(df["issue_d"])

    train_df = df[df["issue_d"] <  VAL_START]
    val_df   = df[(df["issue_d"] >= VAL_START) & (df["issue_d"] < TEST_START)]
    print(f"  Train rows: {len(train_df):,}  Val rows: {len(val_df):,}")

    X_train = _build_X(train_df["user_idx"].values, train_df["item_idx"].values,
                       user_features, item_features)
    y_train = train_df["label"].values.astype(np.float32)

    X_val = _build_X(val_df["user_idx"].values, val_df["item_idx"].values,
                     user_features, item_features)
    y_val = val_df["label"].values.astype(np.float32)

    print(f"  X_train {X_train.shape}   X_val {X_val.shape}")
    print(f"  Positive rate  train={y_train.mean():.3f}  val={y_val.mean():.3f}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "max_depth":        max_depth,
        "eta":              lr,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method":      "hist",
        "seed":             42,
    }

    print("[train_xgboost] Training …")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50,
        early_stopping_rounds=20,
    )

    val_preds = model.predict(dval)
    auc = roc_auc_score(y_val, val_preds)
    print(f"\n[train_xgboost] Val AUC = {auc:.4f}")
    print(f"  Best iteration: {model.best_iteration}")

    model.save_model(MODEL_PATH)
    print(f"[train_xgboost] Saved → {MODEL_PATH}")

    # ── Update feature_meta.json ─────────────────────────────────────────────
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["repay_feat_dim"] = 1
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print("[train_xgboost] Updated feature_meta.json  repay_feat_dim=1")

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost repayment predictor")
    parser.add_argument("--n-estimators", type=int,   default=300)
    parser.add_argument("--max-depth",    type=int,   default=6)
    parser.add_argument("--lr",           type=float, default=0.1)
    args = parser.parse_args()
    train(n_estimators=args.n_estimators, max_depth=args.max_depth, lr=args.lr)
