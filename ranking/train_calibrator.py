"""
ranking/train_calibrator.py
---------------------------
Train an Isotonic Regression explicitly on the val_df interactions
to calibrate XGBoost raw outputs into true empirical probabilities.
"""
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

PROCESSED_DIR = os.path.join("data", "processed")
SAVED_DIR     = os.path.join("models", "saved")
MODEL_PATH    = os.path.join(SAVED_DIR, "xgboost_repay.json")
CALIBRATOR_PATH = os.path.join(SAVED_DIR, "repay_calibrator.pkl")

VAL_START  = "2017-01-01"
TEST_START = "2018-01-01"


def _build_X(user_idxs: np.ndarray, item_idxs: np.ndarray,
             user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
    return np.hstack([user_features[user_idxs], item_features[item_idxs]])


def train_calibrator():
    uf_path = os.path.join(PROCESSED_DIR, "user_features.npy")
    if_path = os.path.join(PROCESSED_DIR, "item_features.npy")
    user_features = np.load(uf_path)
    item_features = np.load(if_path)
    
    interactions_path = os.path.join(PROCESSED_DIR, "interactions_all.parquet")
    df = pd.read_parquet(interactions_path)
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    
    val_df = df[(df["issue_d"] >= VAL_START) & (df["issue_d"] < TEST_START)]
    print(f"Loaded Val DF: {len(val_df)} rows for calibration.")
    
    X_val = _build_X(val_df["user_idx"].values, val_df["item_idx"].values,
                     user_features, item_features)
    y_val = val_df["label"].values.astype(np.float32)
    
    # Generate raw XGBoost predictions (logits/uncalibrated probabilities)
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    dval = xgb.DMatrix(X_val, label=y_val)
    raw_preds = model.predict(dval).astype(np.float32)
    
    print(f"Raw brier score: {brier_score_loss(y_val, raw_preds):.4f}")
    
    # Train Calibrator
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(raw_preds, y_val)
    
    calibrated_preds = calibrator.predict(raw_preds)
    print(f"Calibrated brier score: {brier_score_loss(y_val, calibrated_preds):.4f}")
    
    # Evaluate calibration curve
    print(f"Calibrator maps raw prob 0.2 -> {calibrator.predict([0.2])[0]:.4f}")
    print(f"Calibrator maps raw prob 0.5 -> {calibrator.predict([0.5])[0]:.4f}")
    print(f"Calibrator maps raw prob 0.8 -> {calibrator.predict([0.8])[0]:.4f}")
    
    # Save calibrator using standard pickle
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(calibrator, f)
        
    print(f"Saved calibrator to {CALIBRATOR_PATH}")


if __name__ == "__main__":
    train_calibrator()
