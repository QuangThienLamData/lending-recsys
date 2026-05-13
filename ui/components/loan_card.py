"""
ui/components/loan_card.py
Renders a single recommended loan product card.

Edit here to change:
  - Grade colors  (GRADE_COLOR)
  - Card layout   (metric columns, progress bar position)
  - Copy / labels on each card field
"""

import streamlit as st

# ── Grade color mapping ────────────────────────────────────────────────────────
# Keys are LC loan grades A–G; values are hex background colors for the badge.
GRADE_COLOR: dict[str, str] = {
    "A": "#059669",
    "B": "#10B981",
    "C": "#F59E0B",
    "D": "#F97316",
    "E": "#EF4444",
    "F": "#DC2626",
    "G": "#991B1B",
}


def render_loan_card(item: dict) -> None:
    """
    Render one loan product recommendation as a bordered card.

    Expected keys in `item`:
        rank, item_id, grade, int_rate, loan_amnt, term, purpose,
        score, xgb_repay_prob (optional), positive_rate
    """
    grade      = item["grade"]
    color      = GRADE_COLOR.get(grade, "#6B7280")
    repay_prob = item.get("xgb_repay_prob", item["positive_rate"])
    repay_pct  = int(repay_prob * 100)
    repay_color = (
        "#059669" if repay_pct >= 70
        else "#F59E0B" if repay_pct >= 50
        else "#EF4444"
    )

    with st.container(border=True):
        # ── Card header: rank + ID + grade badge ──────────────────────────────
        col_info, col_score = st.columns([5, 1])

        with col_info:
            st.markdown(
                f"<span style='font-size:0.8rem;color:#9CA3AF;'>#{item['rank']}</span>"
                f"&nbsp;&nbsp;"
                f"<strong style='font-size:1.02rem;'>{item['item_id']}</strong>"
                f"&nbsp;&nbsp;"
                f"<span style='"
                f"  background:{color};color:white;"
                f"  padding:3px 10px;border-radius:6px;"
                f"  font-weight:700;font-size:0.82rem;"
                f"'>Grade {grade}</span>",
                unsafe_allow_html=True,
            )

        with col_score:
            st.markdown(
                f"<div style='text-align:right;'>"
                f"  <span style='font-size:0.72rem;color:#9CA3AF;text-transform:uppercase;"
                f"              letter-spacing:0.06em;'>Match Score</span><br>"
                f"  <strong style='font-size:1.05rem;'>{item['score']:.4f}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Key metrics row ────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Interest Rate",    f"{item['int_rate']:.1f}%")
        c2.metric("Avg Loan Amount",  f"${item['loan_amnt']:,.0f}")
        c3.metric("Term",             item["term"])
        c4.metric("Purpose",          item["purpose"].replace("_", " ").title())

        # ── Repayment probability bar ──────────────────────────────────────────
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        st.progress(
            repay_pct / 100,
            text=f"Repayment Probability: **{repay_pct}%**",
        )
