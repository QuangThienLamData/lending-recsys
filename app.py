"""
app.py  —  Loan Approval Portal (Streamlit UI)
Run with:  streamlit run app.py
"""

import os
import hashlib
import pickle
import pandas as pd
import numpy as np
import requests
import streamlit as st
import faiss
import datetime
from dotenv import load_dotenv

from api.schemas import RecommendRequest
from api.recommender import run_recommendation_pipeline

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

@st.cache_resource(show_spinner="Loading models into memory...")
def load_artefacts():
    artefacts = {}
    MODEL_DIR         = os.getenv("MODEL_DIR",          "models/saved")
    FAISS_INDEX_PATH  = os.getenv("FAISS_INDEX_PATH",   os.path.join(MODEL_DIR, "faiss.index"))
    RANKING_MODEL_PATH= os.getenv("RANKING_MODEL_PATH", os.path.join(MODEL_DIR, "ranking_model.pt"))
    ENCODERS_PATH     = os.getenv("ENCODERS_PATH",      os.path.join(MODEL_DIR, "encoders.pkl"))
    META_PATH         = os.path.join("data", "processed", "feature_meta.json")
    ITEM_LOOKUP_PATH  = os.path.join("data", "processed", "item_lookup.csv")
    USER_PROFILE_PATH = os.path.join("data", "processed", "user_profile_lookup.parquet")
    USER_EMB_PATH     = os.path.join(MODEL_DIR, "als_user_embeddings.npy")
    ITEM_EMB_PATH     = os.path.join(MODEL_DIR, "als_item_embeddings.npy")

    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    artefacts["user_enc"]         = enc["user_enc"]
    artefacts["item_enc"]         = enc["item_enc"]
    artefacts["user_transformer"] = enc.get("user_transformer")
    artefacts["item_transformer"] = enc.get("item_transformer")
    artefacts["top_states"]       = enc.get("top_states", [])

    artefacts["user_emb"] = np.load(USER_EMB_PATH)
    artefacts["item_emb"] = np.load(ITEM_EMB_PATH)
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

artefacts = load_artefacts()

PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "medical",
    "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
]

# ── Name registry ─────────────────────────────────────────────────────────────
KNOWN_NAMES = {
    "100001137": "Vo Quang Thien",
}
_FIRST = [
    "Nguyen Van", "Tran Thi", "Le Minh", "Pham Duc", "Hoang Anh",
    "Do Thi", "Bui Van", "Ngo Thi", "Vo Minh", "Dang Quoc",
    "John", "Jane", "Michael", "Sarah", "David",
    "Emma", "James", "Olivia", "Robert", "Linda",
]
_LAST = [
    "An", "Binh", "Cuong", "Dung", "Hai", "Hoa", "Hung", "Lan",
    "Long", "Mai", "Nam", "Phuong", "Quang", "Son", "Thanh", "Thu",
    "Smith", "Johnson", "Williams", "Brown", "Jones",
]

def _get_name(user_id: str) -> str:
    if user_id in KNOWN_NAMES:
        return KNOWN_NAMES[user_id]
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h >> 8) % len(_LAST)]}"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Portal",
    page_icon="🏦",
    layout="wide",
)

# ── Navigation Injection ──────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Loan Application", "Internal Staff Dashboard"])

if page == "Internal Staff Dashboard":
    st.title("🛡️ Internal Staff Dashboard")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
        
    if not st.session_state["admin_authenticated"]:
        st.subheader("Login Required")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if user == "admin" and pwd == "admin":
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    if st.session_state["admin_authenticated"]:
        if st.button("Logout"):
            st.session_state["admin_authenticated"] = False
            st.rerun()
            
        st.subheader("Customer Query History (Simulation)")
        data_path = os.path.join("data", "simulation_data.csv")
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                # Sort by timestamp descending
                if "Timestamp" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                    df = df.sort_values(by="Timestamp", ascending=False)
                    
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Queries", len(df))
                c2.metric("Total Loan Amount Requested", f"${df['Loan Amount'].sum():,.0f}")
                c3.metric("Avg Expected Interest Rate", f"{df['Expected Interest Rate'].mean():.2f}%")
                
                req_approved = df['Estimated Repay Score'] > 0.60
                acceptance_rate = req_approved.mean() if len(df) > 0 else 0
                c4.metric("Avg Acceptance Rate (>60% Score)", f"{acceptance_rate:.1%}")
                
                st.markdown("**Recent Queries**")
                st.dataframe(df.head(50), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Advanced Analytics")
                
                # ── Timeline ──
                if "Timestamp" in df.columns:
                    st.markdown("**Query Volume Over Time**")
                    df["Date"] = df["Timestamp"].dt.date
                    daily_counts = df.groupby("Date").size()
                    st.line_chart(daily_counts)
                
                # ── Two column charts ──
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Query Purposes Breakdown**")
                    purpose_counts = df["Purpose"].value_counts().reset_index()
                    purpose_counts.columns = ["Purpose", "Count"]
                    st.bar_chart(purpose_counts.set_index("Purpose"))
                    
                with col2:
                    st.markdown("**Average Repay Score by Purpose**")
                    score_by_pup = df.groupby("Purpose")["Estimated Repay Score"].mean()
                    st.bar_chart(score_by_pup)
            except Exception as e:
                st.error(f"Error loading simulation data: {e}")
        else:
            st.warning("Simulation data not found.")
    st.stop()

st.title("🏦 Loan Approval Portal")
st.caption("Powered by ALS + FAISS + NeuMF + XGBoost Repayment Scoring")

# ── Health check ──────────────────────────────────────────────────────────────
if artefacts:
    st.sidebar.success("Models loaded locally")
else:
    st.sidebar.error("Failed to load models")

# ── Sidebar — advanced settings ───────────────────────────────────────────────
with st.sidebar:
    st.header("Advanced Settings")
    top_k          = st.slider("Loan products to show", 1, 20, 5)
    retrieval_pool = st.slider("Candidate pool size", 10, 200, 100,
                               help="Candidates evaluated before ranking")
    use_llm        = st.toggle("GPT reranking", value=False)
    llm_pool       = st.slider("LLM candidate pool", 1, 50, 20,
                               help="How many top items from the ranking model the LLM sees before picking the final results",
                               disabled=not use_llm)


# ── Main panel — application form ─────────────────────────────────────────────
st.subheader("Loan Application")

col_a, col_b = st.columns(2)
with col_a:
    identity_id = st.text_input("Identity ID", value="100001137",
                                help="Your national identity / member ID")
with col_b:
    default_name = _get_name(identity_id.strip()) if identity_id.strip() else ""
    applicant_name = st.text_input("Full Name", value=default_name)

quick_recs = False
if identity_id.strip() and applicant_name.strip():
    quick_recs = st.button("Some Suitable Loaning for you", use_container_width=True)

st.markdown("**Loan Request**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=40000.0,
                                value=10000.0, step=500.0)
with col2:
    int_rate = st.number_input("Expected Interest Rate (%)", min_value=1.0, max_value=40.0,
                               value=12.0, step=0.5)
with col3:
    purpose = st.selectbox("Purpose", PURPOSES,
                           index=PURPOSES.index("debt_consolidation"))
with col4:
    term = st.selectbox("Term", ["36 months", "60 months"])

# ── Ask AI Prompt ─────────────────────────────────────────────────────────────
ai_prompt = ""
with st.expander("Ask AI to give Personal Loaning Recommendation 🤖"):
    st.info("Input your current financial need, and our AI will retrieve the most beneficial relevant loans for you.")
    ai_prompt = st.text_area("Your Scenario / Request:", height=80,
                             placeholder="e.g. I want to buy a car with price $50000 but now I only have $10000...")
    if ai_prompt.strip():
        use_llm = True

submit = st.button("Check Approval", type="primary", use_container_width=True)

# ── Detect setting changes that require re-fetching recommendations ───────────
# Auto-resubmit when GPT reranking toggle or llm_pool changes, using the
# previously submitted form values so the user doesn't have to click again.
auto_resubmit = False
if not submit and not quick_recs and "result" in st.session_state:
    prev = st.session_state["result"].get("settings", {})
    if (prev.get("use_llm") != use_llm or 
        prev.get("llm_pool") != llm_pool or
        prev.get("user_prompt", "") != ai_prompt.strip()):
        auto_resubmit = True

# ── Processing ────────────────────────────────────────────────────────────────
if submit or quick_recs or auto_resubmit:
    if auto_resubmit:
        # Reuse the previously submitted form values
        prev_r      = st.session_state["result"]
        _user_id    = prev_r["identity_id"]
        _name       = prev_r["name"]
        _loan_amnt  = prev_r["loan_amnt"]
        _int_rate   = prev_r["int_rate"]
        _purpose    = prev_r["purpose"]
        _term       = prev_r["term"]
        _is_cold    = prev_r["is_cold"]
    else:
        if not identity_id.strip():
            st.warning("Please enter your Identity ID.")
            st.stop()
        _user_id   = identity_id.strip()
        _name      = applicant_name.strip() or _get_name(identity_id.strip())
        _loan_amnt = loan_amnt
        _int_rate  = int_rate
        _purpose   = purpose
        _term      = term
        _is_cold   = None   # will be fetched below

    with st.spinner("Evaluating your application..."):
        payload = {
            "user_id":           _user_id,
            "top_k":             top_k,
            "use_llm_rerank":    use_llm,
            "llm_pool":          llm_pool,
            "retrieval_pool":    retrieval_pool,
            "loan_amnt_request": _loan_amnt,
            "int_rate_request":  _int_rate,
            "purpose_request":   _purpose,
            "term_request":      _term,
            "user_prompt":       ai_prompt.strip() if ai_prompt.strip() else None,
        }

        try:
            req = RecommendRequest(**payload)
            data = run_recommendation_pipeline(req, artefacts)
        except Exception as e:
            st.error(f"Error evaluating application: {e}")
            st.stop()

        if _is_cold is None:
            known = _user_id in set(artefacts["user_enc"].classes_)
            _is_cold = not known

        # Log query to simulation data only on explicit 'Check Approval' submit
        if submit:
            log_path = os.path.join("data", "simulation_data.csv")
            if os.path.exists(log_path):
                try:
                    new_record = pd.DataFrame([{
                        "ID": _user_id,
                        "Full Name": _name,
                        "Loan Amount": _loan_amnt,
                        "Expected Interest Rate": _int_rate,
                        "Purpose": _purpose,
                        "Term": _term,
                        "Estimated Repay Score": data.get("approval_score", 0.0),
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }])
                    new_record.to_csv(log_path, mode='a', header=False, index=False)
                except Exception as e:
                    pass  # Ignore logging errors so we do not disrupt user flow

    # ── Persist everything in session state so reruns keep the result ─────────
    st.session_state["result"] = {
        "data":           data,
        "name":           _name,
        "identity_id":    _user_id,
        "is_cold":        _is_cold,
        "loan_amnt":      _loan_amnt,
        "int_rate":       _int_rate,
        "purpose":        _purpose,
        "term":           _term,
        "approved":       data.get("approved", False),
        "approval_score": data.get("approval_score", 0.0),
        "settings": {
            "use_llm":  use_llm,
            "llm_pool": llm_pool,
            "top_k":    top_k,
            "user_prompt": ai_prompt.strip(),
        },
    }
    if quick_recs:
        st.session_state["quick_recs_mode"] = True
        st.session_state["show_loans"] = True
    elif submit:
        st.session_state["quick_recs_mode"] = False
        st.session_state["show_loans"] = False
    elif auto_resubmit and "quick_recs_mode" not in st.session_state:
        st.session_state["quick_recs_mode"] = False

# ── Render result (works for both submit click and subsequent reruns) ─────────
if "result" in st.session_state:
    r              = st.session_state["result"]
    data           = r["data"]
    name           = r["name"]
    approved       = r["approved"]
    approval_score = r["approval_score"]
    is_cold        = r["is_cold"]
    quick_mode     = st.session_state.get("quick_recs_mode", False)

    st.divider()

    if not quick_mode:
        # ── Applicant summary ─────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Applicant",      name)
        c2.metric("Identity ID",    r["identity_id"])
        c3.metric("Applicant Type", "New" if is_cold else "Returning")
        c4.metric("Loan Requested", f"${r['loan_amnt']:,.0f}")
        c5.metric("Repay Score",    f"{approval_score:.1%}")
    
        st.caption(f"Purpose: {r['purpose'].replace('_', ' ').title()}  |  "
                   f"Term: {r['term']}  |  "
                   f"Rate: {r['int_rate']:.1f}%  |  "
                   f"Pipeline: {' → '.join(data['pipeline_stages'])}")
        st.divider()
    
        # ── Approval result banner ────────────────────────────────────────────────
        if approved:
            st.markdown(
                f"""
                <div style="background:#1e8449;color:white;padding:28px 32px;
                            border-radius:12px;text-align:center;margin-bottom:20px;">
                    <h1 style="margin:0;font-size:2.6rem;">APPROVED</h1>
                    <p style="margin:10px 0 0 0;font-size:1.1rem;">
                        Congratulations, <strong>{name}</strong>!
                        Your loan application has been approved with a repayment score of
                        <strong>{approval_score:.1%}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            reason = (
                f"Predicted repayment score ({approval_score:.1%}) is below the required threshold (65%)."
            )
            st.markdown(
                f"""
                <div style="background:#922b21;color:white;padding:28px 32px;
                            border-radius:12px;text-align:center;margin-bottom:20px;">
                    <h1 style="margin:0;font-size:2.6rem;">NOT APPROVED</h1>
                    <p style="margin:10px 0 0 0;font-size:1.1rem;">{reason}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.get("show_loans") or quick_mode or not approved:
        st.subheader("Recommended Loan Products for " + name)

        GRADE_COLOR = {
            "A": "#1e8449", "B": "#27ae60",
            "C": "#f39c12", "D": "#e67e22",
            "E": "#e74c3c", "F": "#c0392b", "G": "#922b21",
        }

        for item in data["recommendations"]:
            grade   = item["grade"]
            color   = GRADE_COLOR.get(grade, "#888")
            xgb_pct = int(item.get("xgb_repay_prob", item["positive_rate"]) * 100)

            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(
                        f"**#{item['rank']}  {item['item_id']}**  "
                        f"<span style='background:{color};color:white;"
                        f"padding:2px 8px;border-radius:4px;font-weight:bold'>"
                        f"Grade {grade}</span>",
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Interest Rate",   f"{item['int_rate']:.1f}%")
                    c2.metric("Avg Loan Amount", f"${item['loan_amnt']:,.0f}")
                    c3.metric("Term",            item["term"])
                    c4.metric("Purpose",         item["purpose"].replace("_", " ").title())
                with right:
                    st.metric("Match Score", f"{item['score']:.4f}")
                    st.progress(xgb_pct, text=f"Repay Probability {xgb_pct}%")



    # ── SHAP explanation + improvement suggestions ─────────────────────────────
    shap_features = data.get("shap_features", [])
    improvements  = data.get("improvements",  [])
    user_profile  = data.get("user_profile",  {})
    llm_advice    = data.get("llm_advice",    "")

    if not quick_mode and (shap_features or user_profile):
        st.divider()
        st.subheader("Repayment Score Analysis")

        col_profile, col_shap = st.columns([1, 1])

        # ── Left: current user profile ─────────────────────────────────────
        with col_profile:
            st.markdown("**Your Financial Profile**")
            profile_rows = [
                ("FICO Score",      f"{user_profile.get('fico_range_low','?')} – {user_profile.get('fico_range_high','?')}"),
                ("Annual Income",   f"${user_profile.get('annual_inc', 0):,.0f}" if user_profile.get('annual_inc') else "N/A"),
                ("DTI Ratio",       f"{user_profile.get('dti', '?')}%"),
                ("Home Ownership",  str(user_profile.get("home_ownership", "N/A")).title()),
                ("State",           str(user_profile.get("addr_state", "N/A"))),
            ]
            for label, val in profile_rows:
                c_l, c_r = st.columns([2, 1])
                c_l.markdown(f"<span style='color:#aaa'>{label}</span>", unsafe_allow_html=True)
                c_r.markdown(f"**{val}**")

        # ── Right: SHAP bar chart ───────────────────────────────────────────
        with col_shap:
            if shap_features:
                st.markdown("**Feature Impact on Repay Score**")
                shap_df = pd.DataFrame(shap_features)[["label", "shap_value"]]
                shap_df = shap_df.set_index("label").sort_values("shap_value")

                # Build colored HTML bar chart
                max_abs = max(abs(shap_df["shap_value"].max()),
                              abs(shap_df["shap_value"].min()), 1e-6)
                bars_html = ""
                for feat_label, row in shap_df.iterrows():
                    sv       = row["shap_value"]
                    pct      = abs(sv) / max_abs * 100
                    color    = "#27ae60" if sv >= 0 else "#e74c3c"
                    sign     = "+" if sv >= 0 else ""
                    align    = "left" if sv >= 0 else "right"
                    bars_html += (
                        f"<div style='margin:4px 0;font-size:0.85rem;'>"
                        f"<span style='display:inline-block;width:160px;color:#ddd'>{feat_label}</span>"
                        f"<span style='display:inline-block;width:{pct:.0f}%;max-width:55%;height:14px;"
                        f"background:{color};border-radius:3px;vertical-align:middle'></span>"
                        f"<span style='margin-left:6px;color:{color};font-weight:bold'>{sign}{sv:.3f}</span>"
                        f"</div>"
                    )
                st.markdown(bars_html, unsafe_allow_html=True)
                st.caption("Positive = helps approval  |  Negative = hurts approval")

        # ── Improvement suggestions table ───────────────────────────────────
        if improvements and not approved:
            st.markdown("**How to Improve Your Approval Chances**")
            st.caption("Minimum value changes (holding everything else fixed) that would push your score to ≥ 65%. Factors that cannot guarantee approval locally are hidden.")

            for imp in improvements:
                label   = imp["label"]
                cur     = imp["current_value"]
                req     = imp.get("min_required")
                ok      = imp["achievable"]

                if req is None:
                    continue
                elif req == cur:
                    st.markdown(f"- **{label}**: current `{cur}` — already sufficient ✓")
                else:
                    field = imp["field"]
                    if field in ["dti", "loan_amnt"]:
                        arrow, verb = "↓", "reduce to"
                    else:
                        arrow, verb = "↑", "increase to"

                    delta = abs(req - cur)
                    fmt   = f"{req:,.0f}" if field in ["annual_inc", "loan_amnt"] else f"{req:.1f}"
                    cur_fmt = f"{cur:,.0f}" if field in ["annual_inc", "loan_amnt"] else f"{cur:.1f}"
                    st.markdown(
                        f"- **{label}**: current `{cur_fmt}` {arrow} {verb} "
                        f"<span style='color:#27ae60;font-weight:bold'>`{fmt}`</span> "
                        f"(change of {delta:,.1f})",
                        unsafe_allow_html=True,
                    )

        # ── LLM financial advice ───────────────────────────────────────────
        if llm_advice:
            st.markdown("**AI Financial Advisor**")
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border-left:4px solid #4a90d9;
                            padding:16px 20px;border-radius:6px;
                            color:#e0e0e0;font-size:0.95rem;line-height:1.6;">
                    {llm_advice}
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    st.info("Fill in the form above and click **Check Approval** to begin.")
