"""
preprocessing/feature_engineering.py
--------------------------------------
Constructs and normalises user and item feature vectors from the cleaned
interaction DataFrame produced by build_interactions.py.

Fits:
  - StandardScaler on continuous user/item features
  - OneHotEncoder on categorical user/item features
  - Updates encoders.pkl with the fitted transformers

Outputs written to data/processed/:
  user_features.npy   — float32 array (n_users, user_feat_dim)
  item_features.npy   — float32 array (n_items, item_feat_dim)
  feature_meta.json   — feature dimension info for downstream models
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

OUT_DIR      = os.path.join("data", "processed")
ENCODERS_PATH = os.path.join("models", "saved", "encoders.pkl")

# Top-N states to one-hot encode; rest collapsed to "other"
TOP_N_STATES = 10

# ── User feature spec ───────────────────────────────────────────────────────
USER_CONTINUOUS = ["annual_inc", "dti", "fico_range_low", "fico_range_high"]
USER_CATEGORICAL = ["home_ownership"]   # addr_state handled separately

# ── Item feature spec ───────────────────────────────────────────────────────
ITEM_CONTINUOUS  = ["int_rate", "loan_amnt"]
ITEM_CATEGORICAL = ["grade", "purpose", "term"]


def _load_encoders():
    with open(ENCODERS_PATH, "rb") as f:
        return pickle.load(f)


def _save_encoders(enc_dict):
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(enc_dict, f)


def _clip_state(series: pd.Series, top_states) -> pd.Series:
    return series.where(series.isin(top_states), other="other")


def build_user_features(df: pd.DataFrame, user_enc) -> np.ndarray:
    """
    Build one feature row per unique user.
    Continuous features are scaled; categoricals are one-hot encoded.
    """
    # One row per user: take the last recorded loan application attributes
    user_df = (
        df.sort_values("issue_d")
        .groupby("member_id")
        .last()
        .reset_index()
    )
    # Align with user_enc order
    user_df = (
        pd.DataFrame({"member_id": user_enc.classes_})
        .merge(user_df, on="member_id", how="left")
    )

    # Collapse rare states
    top_states = (
        user_df["addr_state"].value_counts().head(TOP_N_STATES).index.tolist()
        if "addr_state" in user_df.columns else []
    )
    if top_states:
        user_df["addr_state"] = _clip_state(user_df["addr_state"].fillna("other"), top_states)
        categorical_cols = USER_CATEGORICAL + ["addr_state"]
    else:
        categorical_cols = USER_CATEGORICAL

    # Fill missing numeric values
    for col in USER_CONTINUOUS:
        if col not in user_df.columns:
            user_df[col] = 0.0
    user_df[USER_CONTINUOUS] = user_df[USER_CONTINUOUS].apply(pd.to_numeric, errors="coerce")

    # Build transformer
    transformer = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), USER_CONTINUOUS),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe",    OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]), categorical_cols),
    ], remainder="drop")

    user_features = transformer.fit_transform(user_df).astype(np.float32)
    print(f"  User feature matrix: {user_features.shape}")
    return user_features, transformer, top_states


def build_item_features(item_lookup: pd.DataFrame) -> np.ndarray:
    """
    Build one feature row per unique item, aligned with item_idx order.
    """
    item_df = item_lookup.sort_values("item_idx").reset_index(drop=True)

    for col in ITEM_CONTINUOUS:
        if col not in item_df.columns:
            item_df[col] = 0.0
    item_df[ITEM_CONTINUOUS] = item_df[ITEM_CONTINUOUS].apply(pd.to_numeric, errors="coerce")

    transformer = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), ITEM_CONTINUOUS),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe",    OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]), ITEM_CATEGORICAL),
    ], remainder="drop")

    item_features = transformer.fit_transform(item_df).astype(np.float32)
    print(f"  Item feature matrix: {item_features.shape}")
    return item_features, transformer


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    enc_dict = _load_encoders()
    user_enc = enc_dict["user_enc"]
    item_enc = enc_dict["item_enc"]

    # Load processed data
    interactions_path = os.path.join(OUT_DIR, "interactions_all.parquet")
    item_lookup_path  = os.path.join(OUT_DIR, "item_lookup.csv")

    print("[feature_engineering] Loading interaction data …")
    df          = pd.read_parquet(interactions_path)
    item_lookup = pd.read_csv(item_lookup_path)

    print("[feature_engineering] Building user features …")
    user_features, user_transformer, top_states = build_user_features(df, user_enc)

    print("[feature_engineering] Building item features …")
    item_features, item_transformer = build_item_features(item_lookup)

    # Save arrays
    np.save(os.path.join(OUT_DIR, "user_features.npy"), user_features)
    np.save(os.path.join(OUT_DIR, "item_features.npy"), item_features)
    print(f"  Saved user_features.npy  {user_features.shape}")
    print(f"  Saved item_features.npy  {item_features.shape}")

    # Persist transformers alongside existing encoders
    enc_dict["user_transformer"] = user_transformer
    enc_dict["item_transformer"] = item_transformer
    enc_dict["top_states"]       = top_states
    _save_encoders(enc_dict)
    print(f"  Updated {ENCODERS_PATH} with fitted transformers")

    # Save feature dimension metadata
    meta = {
        "user_feat_dim": int(user_features.shape[1]),
        "item_feat_dim": int(item_features.shape[1]),
        "n_users":       int(user_features.shape[0]),
        "n_items":       int(item_features.shape[0]),
    }
    with open(os.path.join(OUT_DIR, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved feature_meta.json: {meta}")

    print("\n[feature_engineering] Done.")


if __name__ == "__main__":
    main()
