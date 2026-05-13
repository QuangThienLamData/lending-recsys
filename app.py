"""
app.py — Loan Approval Portal entry point
Run with:  streamlit run app.py

This file is intentionally thin:
  - page config (must be the very first Streamlit call)
  - model loading
  - global CSS injection
  - sidebar (navigation + advanced settings)
  - routing to ui/pages/

All UI logic lives under ui/. See ui/UI_STRUCTURE.md for a full map.
"""

import hashlib
import os
import pickle

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ui.styles import inject_styles
import ui.pages.loan_application as page_loan
import ui.pages.staff_dashboard as page_staff

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="Loan Approval Portal",
    page_icon="🏦",
    layout="wide",
)

inject_styles()


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artefacts() -> dict:
    artefacts: dict = {}

    MODEL_DIR          = os.getenv("MODEL_DIR",          "models/saved")
    FAISS_INDEX_PATH   = os.getenv("FAISS_INDEX_PATH",   os.path.join(MODEL_DIR, "faiss.index"))
    RANKING_MODEL_PATH = os.getenv("RANKING_MODEL_PATH", os.path.join(MODEL_DIR, "ranking_model.pt"))
    ENCODERS_PATH      = os.getenv("ENCODERS_PATH",      os.path.join(MODEL_DIR, "encoders.pkl"))
    META_PATH          = os.path.join("data", "processed", "feature_meta.json")
    ITEM_LOOKUP_PATH   = os.path.join("data", "processed", "item_lookup.csv")
    USER_PROFILE_PATH  = os.path.join("data", "processed", "user_profile_lookup.parquet")
    USER_EMB_PATH      = os.path.join(MODEL_DIR, "als_user_embeddings.npy")
    ITEM_EMB_PATH      = os.path.join(MODEL_DIR, "als_item_embeddings.npy")

    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    artefacts["user_enc"]         = enc["user_enc"]
    artefacts["item_enc"]         = enc["item_enc"]
    artefacts["user_transformer"] = enc.get("user_transformer")
    artefacts["item_transformer"] = enc.get("item_transformer")
    artefacts["top_states"]       = enc.get("top_states", [])

    artefacts["user_emb"]    = np.load(USER_EMB_PATH)
    artefacts["item_emb"]    = np.load(ITEM_EMB_PATH)
    artefacts["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)

    from ranking.predictor import RankingPredictor
    artefacts["ranking_predictor"] = RankingPredictor(
        model_path=RANKING_MODEL_PATH,
        meta_path=META_PATH,
        device="cpu",
    )

    artefacts["item_lookup"] = pd.read_csv(ITEM_LOOKUP_PATH)
    if os.path.exists(USER_PROFILE_PATH):
        artefacts["user_profile_lookup"] = pd.read_parquet(USER_PROFILE_PATH)

    return artefacts


# ── App-level constants ────────────────────────────────────────────────────────
PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase",     "small_business", "car",           "medical",
    "moving",             "vacation",       "house",         "wedding",
    "renewable_energy",   "educational",
]

_KNOWN_NAMES: dict[str, str] = {
    "100001137": "Vo Quang Thien",
}
_FIRST = [
    "Nguyen Van", "Tran Thi",  "Le Minh",   "Pham Duc",  "Hoang Anh",
    "Do Thi",     "Bui Van",   "Ngo Thi",   "Vo Minh",   "Dang Quoc",
    "John",       "Jane",      "Michael",   "Sarah",     "David",
    "Emma",       "James",     "Olivia",    "Robert",    "Linda",
]
_LAST = [
    "An",     "Binh",    "Cuong",    "Dung",     "Hai",
    "Hoa",    "Hung",    "Lan",      "Long",     "Mai",
    "Nam",    "Phuong",  "Quang",    "Son",      "Thanh",
    "Thu",    "Smith",   "Johnson",  "Williams", "Brown", "Jones",
]


def _get_name(user_id: str) -> str:
    if user_id in _KNOWN_NAMES:
        return _KNOWN_NAMES[user_id]
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h >> 8) % len(_LAST)]}"


# ── Load models (cached) ───────────────────────────────────────────────────────
artefacts = load_artefacts()

# ── Sidebar ────────────────────────────────────────────────────────────────────
top_k = retrieval_pool = use_llm = llm_pool = None  # defaults before sidebar renders

with st.sidebar:
    st.markdown("## Loan Portal")

    # Model status indicator
    if artefacts:
        st.success("Models ready")
    else:
        st.error("Models failed to load")

    st.divider()

    page = st.radio(
        "Navigate to",
        ["Loan Application", "Staff Dashboard"],
        label_visibility="collapsed",
    )

    if page == "Loan Application":
        st.divider()
        with st.expander("Advanced Settings"):
            top_k = st.slider(
                "Results to show", 1, 20, 5,
                help="Number of loan products displayed in the recommendations list",
            )
            retrieval_pool = st.slider(
                "Candidate pool", 10, 200, 100,
                help="Candidates evaluated by the ranking model before final selection",
            )
            use_llm = st.toggle("GPT re-ranking", value=False)
            llm_pool = st.slider(
                "LLM pool size", 1, 50, 20,
                disabled=not use_llm,
                help="How many top-ranked items the LLM sees before picking the final list",
            )

# Apply defaults if the Loan Application sidebar was not rendered
settings = {
    "top_k":          top_k          if top_k          is not None else 5,
    "retrieval_pool": retrieval_pool if retrieval_pool is not None else 100,
    "use_llm":        use_llm        if use_llm        is not None else False,
    "llm_pool":       llm_pool       if llm_pool       is not None else 20,
}

# ── Route ──────────────────────────────────────────────────────────────────────
if page == "Staff Dashboard":
    page_staff.render()
else:
    st.title("Loan Approval Portal")
    page_loan.render(artefacts, PURPOSES, _get_name, settings)
