"""
ui/components/analysis_panel.py
Renders the post-decision analysis section:
  - render_applicant_summary  : 5-metric row + pipeline caption
  - render_profile            : user financial profile table
  - render_shap_chart         : horizontal SHAP bar chart
  - render_improvements       : actionable improvement suggestions
  - render_llm_advice         : AI advisor panel

Edit here to change:
  - Profile fields shown
  - SHAP chart bar colors / width
  - Improvement suggestion copy
  - LLM advice panel styling
"""

import streamlit as st
import pandas as pd


# ── Applicant summary ─────────────────────────────────────────────────────────

def render_applicant_summary(r: dict, pipeline_stages: list) -> None:
    """
    Show a 5-metric row summarising the applicant and their request,
    followed by a caption with purpose / term / rate / pipeline.
    """
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Applicant",     r["name"])
    c2.metric("Customer ID",   r["identity_id"])
    c3.metric("Customer Type", "New" if r["is_cold"] else "Returning")
    c4.metric("Loan Amount",   f"${r['loan_amnt']:,.0f}")
    c5.metric("Repay Score",   f"{r['approval_score']:.1%}")

    st.caption(
        f"Purpose: **{r['purpose'].replace('_', ' ').title()}**  ·  "
        f"Term: **{r['term']}**  ·  "
        f"Rate: **{r['int_rate']:.1f}%**  ·  "
        f"Pipeline: {' → '.join(pipeline_stages)}"
    )


# ── Financial profile ─────────────────────────────────────────────────────────

def render_profile(user_profile: dict) -> None:
    """Display the user's stored financial profile as a two-column list."""
    st.markdown("**Your Financial Profile**")
    rows = [
        ("FICO Score",     f"{user_profile.get('fico_range_low', '?')} – "
                           f"{user_profile.get('fico_range_high', '?')}"),
        ("Annual Income",  (f"${user_profile.get('annual_inc', 0):,.0f}"
                            if user_profile.get("annual_inc") else "N/A")),
        ("DTI Ratio",      f"{user_profile.get('dti', '?')}%"),
        ("Home Ownership", str(user_profile.get("home_ownership", "N/A")).title()),
        ("State",          str(user_profile.get("addr_state", "N/A"))),
    ]
    for label, val in rows:
        cl, cr = st.columns([2, 1])
        cl.markdown(
            f"<span style='color:#9CA3AF;font-size:0.88rem;'>{label}</span>",
            unsafe_allow_html=True,
        )
        cr.markdown(f"**{val}**")


# ── SHAP bar chart ────────────────────────────────────────────────────────────

def render_shap_chart(shap_features: list) -> None:
    """
    Render a horizontal HTML bar chart showing each feature's SHAP contribution.
    Green bars = positive impact on approval; red = negative.
    """
    st.markdown("**Feature Impact on Repayment Score**")

    shap_df = (
        pd.DataFrame(shap_features)[["label", "shap_value"]]
        .set_index("label")
        .sort_values("shap_value")
    )
    max_abs = max(abs(shap_df["shap_value"].max()),
                  abs(shap_df["shap_value"].min()), 1e-6)

    bars_html = "<div style='font-size:0.83rem;'>"
    for feat_label, row in shap_df.iterrows():
        sv    = row["shap_value"]
        pct   = abs(sv) / max_abs * 100
        color = "#059669" if sv >= 0 else "#EF4444"
        sign  = "+" if sv >= 0 else ""
        bars_html += (
            f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;'>"
            f"  <span style='"
            f"    width:150px;flex-shrink:0;color:#D1D5DB;"
            f"    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
            f"  ' title='{feat_label}'>{feat_label}</span>"
            f"  <span style='"
            f"    display:inline-block;"
            f"    width:{pct:.0f}%;max-width:50%;min-width:3px;"
            f"    height:12px;background:{color};border-radius:4px;"
            f"  '></span>"
            f"  <span style='color:{color};font-weight:600;white-space:nowrap;'>"
            f"    {sign}{sv:.3f}"
            f"  </span>"
            f"</div>"
        )
    bars_html += "</div>"

    st.markdown(bars_html, unsafe_allow_html=True)
    st.caption("Green = boosts approval  ·  Red = reduces approval chances")


# ── Improvement suggestions ────────────────────────────────────────────────────

def render_improvements(improvements: list) -> None:
    """
    List the minimum per-feature changes that would push the repay score
    above the 65% approval threshold.
    """
    st.markdown("**How to Improve Your Chances**")
    st.caption(
        "Minimum changes (holding everything else fixed) that would push "
        "your score above the 65% approval threshold."
    )

    for imp in improvements:
        req = imp.get("min_required")
        if req is None:
            continue
        cur   = imp["current_value"]
        label = imp["label"]
        field = imp["field"]

        if req == cur:
            st.markdown(f"- **{label}**: `{cur}` — already at a good level ✓")
        else:
            arrow, verb = (
                ("↓", "reduce to") if field in ["dti", "loan_amnt"]
                else ("↑", "increase to")
            )
            delta   = abs(req - cur)
            fmt     = f"{req:,.0f}" if field in ["annual_inc", "loan_amnt"] else f"{req:.1f}"
            cur_fmt = f"{cur:,.0f}" if field in ["annual_inc", "loan_amnt"] else f"{cur:.1f}"
            st.markdown(
                f"- **{label}**: `{cur_fmt}` {arrow} {verb} "
                f"<span style='color:#059669;font-weight:600;'>`{fmt}`</span>"
                f" (change by {delta:,.1f})",
                unsafe_allow_html=True,
            )


# ── LLM advice panel ──────────────────────────────────────────────────────────

def render_llm_advice(llm_advice: str) -> None:
    """Render the GPT-generated financial advice in a styled callout box."""
    st.markdown("**AI Financial Advisor**")
    st.markdown(
        f"""
        <div style="
            background   : rgba(37, 99, 235, 0.07);
            border-left  : 4px solid #2563EB;
            padding      : 16px 20px;
            border-radius: 0 8px 8px 0;
            font-size    : 0.95rem;
            line-height  : 1.75;
            margin-top   : 8px;
        ">
            {llm_advice}
        </div>
        """,
        unsafe_allow_html=True,
    )
