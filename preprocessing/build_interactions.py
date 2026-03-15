"""
preprocessing/build_interactions.py
------------------------------------
Loads the raw LendingClub CSV, defines users and synthetic item IDs,
builds an implicit interaction matrix, and performs a time-based
train / val / test split.

Outputs written to data/processed/:
  train_interactions.npz  — scipy sparse matrix (COO), train split
  val_interactions.npz    — scipy sparse matrix (COO), val split
  test_interactions.npz   — scipy sparse matrix (COO), test split
  item_lookup.csv         — item_idx → (grade, purpose, term, int_rate, …)
  user_id_map.csv         — user_idx ↔ member_id
"""

import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import pickle

# --------------------------------------------------------------------------- #
# Configurable constants
# --------------------------------------------------------------------------- #
RAW_CSV = os.path.join("data", "raw", "accepted_2007_to_2018Q4.csv")
OUT_DIR = os.path.join("data", "processed")
ENCODERS_PATH = os.path.join("models", "saved", "encoders.pkl")

KEEP_COLS = [
    "member_id", "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "purpose", "loan_status", "issue_d",
    "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "home_ownership", "addr_state",
]

POSITIVE_STATUSES = {"Fully Paid", "Current"}
NEGATIVE_STATUSES = {"Charged Off", "Default"}

VAL_START  = "2017-01-01"
TEST_START = "2018-01-01"


def load_raw(path: str) -> pd.DataFrame:
    print(f"[build_interactions] Loading {path} …")
    df = pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows without a usable user id
    # LendingClub CSVs sometimes use 'id' as the borrower proxy when member_id is absent
    if df["member_id"].isna().all():
        print("  member_id column is fully null — using 'id' as user proxy")
        df = pd.read_csv(RAW_CSV, usecols=KEEP_COLS + ["id"], low_memory=False)
        df["member_id"] = df["id"].astype(str)

    df = df.dropna(subset=["member_id", "loan_status", "issue_d"])
    df["member_id"] = df["member_id"].astype(str)

    # Parse dates
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])

    # Keep only signals we can interpret
    mask = df["loan_status"].isin(POSITIVE_STATUSES | NEGATIVE_STATUSES)
    df = df[mask].copy()

    # Binary label: 1 = repaid, 0 = defaulted
    df["label"] = df["loan_status"].isin(POSITIVE_STATUSES).astype(np.int8)

    # Synthetic item ID = grade_purpose_term (e.g. "B_debt_consolidation_36 months")
    df["grade"]   = df["grade"].fillna("Unknown")
    df["purpose"] = df["purpose"].fillna("other")
    df["term"]    = df["term"].str.strip()
    df["item_id"] = df["grade"] + "_" + df["purpose"] + "_" + df["term"]

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Unique users: {df['member_id'].nunique():,}")
    print(f"  Unique items: {df['item_id'].nunique():,}")
    return df


def encode_ids(df: pd.DataFrame):
    user_enc = LabelEncoder().fit(df["member_id"])
    item_enc = LabelEncoder().fit(df["item_id"])
    df = df.copy()
    df["user_idx"] = user_enc.transform(df["member_id"])
    df["item_idx"] = item_enc.transform(df["item_id"])
    return df, user_enc, item_enc


def time_split(df: pd.DataFrame):
    train = df[df["issue_d"] <  VAL_START]
    val   = df[(df["issue_d"] >= VAL_START) & (df["issue_d"] < TEST_START)]
    test  = df[df["issue_d"] >= TEST_START]
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


def to_sparse(df: pd.DataFrame, n_users: int, n_items: int,
              positive_only: bool = True) -> sp.coo_matrix:
    """Build a user × item sparse matrix. Only positive interactions by default."""
    if positive_only:
        df = df[df["label"] == 1]
    rows = df["user_idx"].values
    cols = df["item_idx"].values
    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items))


def build_item_lookup(df: pd.DataFrame, item_enc: LabelEncoder) -> pd.DataFrame:
    """Aggregated item attribute table: one row per item_idx."""
    item_df = (
        df.groupby("item_id")
        .agg(
            int_rate=("int_rate", "mean"),
            loan_amnt=("loan_amnt", "mean"),
            grade=("grade", "first"),
            purpose=("purpose", "first"),
            term=("term", "first"),
            n_loans=("label", "count"),
            positive_rate=("label", "mean"),
        )
        .reset_index()
    )
    item_df["item_idx"] = item_enc.transform(item_df["item_id"])
    item_df = item_df.sort_values("item_idx").reset_index(drop=True)
    return item_df


def main(raw_path: str = RAW_CSV):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)

    df = load_raw(raw_path)
    df = clean(df)
    df, user_enc, item_enc = encode_ids(df)

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    train_df, val_df, test_df = time_split(df)

    # Build sparse matrices (positive interactions only)
    train_mat = to_sparse(train_df, n_users, n_items)
    val_mat   = to_sparse(val_df,   n_users, n_items)
    test_mat  = to_sparse(test_df,  n_users, n_items)

    # Save sparse matrices
    sp.save_npz(os.path.join(OUT_DIR, "train_interactions.npz"), train_mat.tocsr())
    sp.save_npz(os.path.join(OUT_DIR, "val_interactions.npz"),   val_mat.tocsr())
    sp.save_npz(os.path.join(OUT_DIR, "test_interactions.npz"),  test_mat.tocsr())
    print(f"  Saved train/val/test interaction matrices → {OUT_DIR}/")

    # Save the full DataFrame (with labels) for negative sampling in ranking
    df.to_parquet(os.path.join(OUT_DIR, "interactions_all.parquet"), index=False)
    print(f"  Saved full interactions parquet → {OUT_DIR}/interactions_all.parquet")

    # Item lookup table
    item_lookup = build_item_lookup(df, item_enc)
    item_lookup.to_csv(os.path.join(OUT_DIR, "item_lookup.csv"), index=False)
    print(f"  Saved item_lookup.csv  ({len(item_lookup)} items)")

    # User ID map
    user_map = pd.DataFrame({
        "user_idx": np.arange(n_users),
        "member_id": user_enc.classes_,
    })
    user_map.to_csv(os.path.join(OUT_DIR, "user_id_map.csv"), index=False)
    print(f"  Saved user_id_map.csv  ({n_users:,} users)")

    # Persist encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({"user_enc": user_enc, "item_enc": item_enc}, f)
    print(f"  Saved encoders → {ENCODERS_PATH}")

    print("\n[build_interactions] Done.")
    return df, user_enc, item_enc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build interaction matrices from LendingClub CSV")
    parser.add_argument("--raw", default=RAW_CSV, help="Path to raw LendingClub CSV")
    args = parser.parse_args()
    main(raw_path=args.raw)
