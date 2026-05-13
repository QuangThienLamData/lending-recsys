"""
ui/pages/loan_application.py
Full render function for the Loan Application page.

Responsibilities:
  - Personal information form
  - Loan details form
  - AI assistant expander
  - Submit / quick-browse buttons
  - Calling the recommendation pipeline
  - Displaying results (banner, loan cards, analysis)

Edit here to change:
  - Form field layout or defaults
  - Button labels or placement
  - Result section ordering
  - Session-state key names
"""

import os
import datetime
from typing import Callable

import pandas as pd
import streamlit as st

from api.schemas import RecommendRequest
from api.recommender import run_recommendation_pipeline
from ui.components.approval_banner import render_approval_banner
from ui.components.loan_card import render_loan_card
from ui.components.analysis_panel import (
    render_applicant_summary,
    render_improvements,
    render_llm_advice,
    render_profile,
    render_shap_chart,
)

_LABEL = "<p class='ui-section-label'>{}</p>"


def render(
    artefacts: dict,
    purposes: list[str],
    get_name_fn: Callable[[str], str],
    settings: dict,
) -> None:
    """
    Render the complete Loan Application page.

    Parameters
    ----------
    artefacts   : dict     Loaded ML model artefacts (from app.py cache).
    purposes    : list     Ordered list of loan purpose strings.
    get_name_fn : callable fn(user_id: str) -> str  — deterministic name lookup.
    settings    : dict     Keys: top_k, retrieval_pool, use_llm, llm_pool.
    """
    top_k          = settings["top_k"]
    retrieval_pool = settings["retrieval_pool"]
    use_llm        = settings["use_llm"]
    llm_pool       = settings["llm_pool"]

    # ── Section 1: Personal Information ───────────────────────────────────────
    st.markdown(_LABEL.format("Personal Information"), unsafe_allow_html=True)

    col_id, col_name = st.columns(2)
    with col_id:
        identity_id = st.text_input(
            "Customer ID",
            value="100001137",
            help="Your national identity number or member ID",
        )
    with col_name:
        default_name   = get_name_fn(identity_id.strip()) if identity_id.strip() else ""
        applicant_name = st.text_input("Full Name", value=default_name)

    # Quick-browse button — shown only when both fields are filled
    quick_recs = False
    if identity_id.strip() and applicant_name.strip():
        quick_recs = st.button(
            "View My Loan Options",
            help="Browse loan products that match your profile — no approval check",
        )

    # ── Section 2: Loan Details ───────────────────────────────────────────────
    st.markdown(_LABEL.format("Loan Details"), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=500.0, max_value=40_000.0, value=10_000.0, step=500.0,
        )
    with col2:
        int_rate = st.number_input(
            "Expected Interest Rate (%)",
            min_value=1.0, max_value=40.0, value=12.0, step=0.5,
        )
    with col3:
        purpose = st.selectbox(
            "Purpose",
            purposes,
            index=purposes.index("debt_consolidation"),
        )
    with col4:
        term = st.selectbox("Term", ["36 months", "60 months"])

    # ── Section 3: AI Assistant (progressive disclosure) ──────────────────────
    ai_prompt = ""
    with st.expander("Ask AI for a Personalised Recommendation"):
        st.info(
            "Describe your financial situation and our AI will surface the most "
            "relevant loan products for you.",
            icon="💡",
        )
        ai_prompt = st.text_area(
            "Your situation:",
            height=80,
            placeholder=(
                "e.g. I need to consolidate $15,000 in credit card debt "
                "at the lowest possible rate…"
            ),
        )
        if ai_prompt.strip():
            use_llm = True  # override sidebar toggle

    # ── Primary CTA ───────────────────────────────────────────────────────────
    submit = st.button("Check My Approval", type="primary", use_container_width=True)

    # ── Auto-resubmit when LLM settings change while results are visible ──────
    auto_resubmit = False
    if not submit and not quick_recs and "result" in st.session_state:
        prev = st.session_state["result"].get("settings", {})
        if (
            prev.get("use_llm")      != use_llm
            or prev.get("llm_pool")  != llm_pool
            or prev.get("user_prompt", "") != ai_prompt.strip()
        ):
            auto_resubmit = True

    # ── Pipeline call ─────────────────────────────────────────────────────────
    if submit or quick_recs or auto_resubmit:
        if auto_resubmit:
            prev_r     = st.session_state["result"]
            _user_id   = prev_r["identity_id"]
            _name      = prev_r["name"]
            _loan_amnt = prev_r["loan_amnt"]
            _int_rate  = prev_r["int_rate"]
            _purpose   = prev_r["purpose"]
            _term      = prev_r["term"]
            _is_cold   = prev_r["is_cold"]
        else:
            if not identity_id.strip():
                st.warning("Please enter your Customer ID to continue.")
                st.stop()
            _user_id   = identity_id.strip()
            _name      = applicant_name.strip() or get_name_fn(identity_id.strip())
            _loan_amnt = loan_amnt
            _int_rate  = int_rate
            _purpose   = purpose
            _term      = term
            _is_cold   = None

        with st.spinner("Evaluating your application…"):
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
                "user_prompt":       ai_prompt.strip() or None,
            }
            try:
                req  = RecommendRequest(**payload)
                data = run_recommendation_pipeline(req, artefacts)
            except Exception as exc:
                st.error(f"Error evaluating application: {exc}")
                st.stop()

            if _is_cold is None:
                _is_cold = _user_id not in set(artefacts["user_enc"].classes_)

            # Log to CSV on explicit "Check My Approval" clicks only
            if submit:
                log_path = os.path.join("data", "simulation_data.csv")
                if os.path.exists(log_path):
                    try:
                        pd.DataFrame([{
                            "ID":                     _user_id,
                            "Full Name":              _name,
                            "Loan Amount":            _loan_amnt,
                            "Expected Interest Rate": _int_rate,
                            "Purpose":                _purpose,
                            "Term":                   _term,
                            "Estimated Repay Score":  data.get("approval_score", 0.0),
                            "Timestamp":              datetime.datetime.now().strftime(
                                                          "%Y-%m-%d %H:%M:%S"
                                                      ),
                        }]).to_csv(log_path, mode="a", header=False, index=False)
                    except Exception:
                        pass

        # Persist result in session state
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
                "use_llm":     use_llm,
                "llm_pool":    llm_pool,
                "top_k":       top_k,
                "user_prompt": ai_prompt.strip(),
            },
        }
        st.session_state["quick_recs_mode"] = quick_recs
        if submit:
            st.session_state["show_loans"] = False
        elif quick_recs:
            st.session_state["show_loans"] = True

    # ── Empty state ───────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        st.markdown(
            "<div style='text-align:center;color:#6B7280;padding:48px 0;font-size:0.95rem;'>"
            "Fill in the form above and click <strong>Check My Approval</strong> to get started."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Results ───────────────────────────────────────────────────────────────
    r          = st.session_state["result"]
    data       = r["data"]
    approved   = r["approved"]
    quick_mode = st.session_state.get("quick_recs_mode", False)
    show_loans = st.session_state.get("show_loans", False)

    st.divider()

    # Applicant summary + decision banner (full approval check only)
    if not quick_mode:
        render_applicant_summary(r, data["pipeline_stages"])
        st.markdown("")
        render_approval_banner(approved, r["name"], r["approval_score"])

        # After approval: offer a button to browse matching products
        if approved and not show_loans:
            if st.button("Browse Matching Loan Products", use_container_width=True):
                st.session_state["show_loans"] = True
                st.rerun()

    # ── Loan product cards ────────────────────────────────────────────────────
    if show_loans or quick_mode or not approved:
        st.markdown(
            _LABEL.format(f"Recommended Loan Products for {r['name']}"),
            unsafe_allow_html=True,
        )
        for item in data["recommendations"]:
            render_loan_card(item)

    # ── Analysis: profile + SHAP + improvements + LLM (full check only) ──────
    if not quick_mode:
        shap_features = data.get("shap_features", [])
        improvements  = data.get("improvements",  [])
        user_profile  = data.get("user_profile",  {})
        llm_advice    = data.get("llm_advice",    "")

        if shap_features or user_profile:
            st.divider()
            st.markdown("### Repayment Score Analysis")
            col_profile, col_shap = st.columns(2)
            with col_profile:
                if user_profile:
                    render_profile(user_profile)
            with col_shap:
                if shap_features:
                    render_shap_chart(shap_features)

        if improvements and not approved:
            st.markdown("")
            render_improvements(improvements)

        if llm_advice:
            st.markdown("")
            render_llm_advice(llm_advice)
