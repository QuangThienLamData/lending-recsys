"""
ui/components/approval_banner.py
Renders the APPROVED / NOT APPROVED decision banner.

Edit here to change:
  - Banner colors or gradient
  - Copy / messaging
  - Layout (icon placement, font sizes)
"""

import streamlit as st


def render_approval_banner(approved: bool, name: str, score: float) -> None:
    """
    Display a full-width colored banner with the loan decision.

    Parameters
    ----------
    approved : bool    Whether the application was approved.
    name     : str     Applicant's display name.
    score    : float   Repayment probability (0.0 – 1.0).
    """
    if approved:
        st.markdown(
            f"""
            <div style="
                background   : linear-gradient(135deg, #064E3B 0%, #059669 100%);
                color        : white;
                padding      : 28px 36px;
                border-radius: 14px;
                text-align   : center;
                margin       : 8px 0 20px 0;
                box-shadow   : 0 4px 24px rgba(5, 150, 105, 0.28);
            ">
                <div style="font-size:0.82rem;letter-spacing:0.14em;opacity:0.8;margin-bottom:6px;">
                    DECISION RESULT
                </div>
                <h1 style="margin:0;font-size:2.6rem;letter-spacing:0.04em;">
                    &#10003;&nbsp;APPROVED
                </h1>
                <p style="margin:12px 0 0;font-size:1.05rem;opacity:0.92;line-height:1.6;">
                    Congratulations, <strong>{name}</strong>!
                    Your repayment score of <strong>{score:.1%}</strong> meets our criteria.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background   : linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%);
                color        : white;
                padding      : 28px 36px;
                border-radius: 14px;
                text-align   : center;
                margin       : 8px 0 20px 0;
                box-shadow   : 0 4px 24px rgba(185, 28, 28, 0.22);
            ">
                <div style="font-size:0.82rem;letter-spacing:0.14em;opacity:0.8;margin-bottom:6px;">
                    DECISION RESULT
                </div>
                <h1 style="margin:0;font-size:2.6rem;letter-spacing:0.04em;">
                    &#10007;&nbsp;NOT APPROVED
                </h1>
                <p style="margin:12px 0 0;font-size:1.05rem;opacity:0.92;line-height:1.6;">
                    Your repayment score (<strong>{score:.1%}</strong>) is below the required
                    threshold of 65%.&nbsp; See the improvement tips below to strengthen
                    your next application.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
