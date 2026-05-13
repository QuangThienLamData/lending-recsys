"""
ui/pages/staff_dashboard.py
Internal Staff Dashboard page — login-gated analytics view.

Sections:
  - Login form (admin / admin)
  - KPI overview (total apps, loan volume, avg rate, approval rate)
  - Recent applications table with column formatting
  - Analytics charts (volume over time, purpose breakdown, score by purpose)

Edit here to change:
  - KPI definitions or thresholds
  - Table columns and formatting
  - Chart types or groupings
  - Login credentials (replace with a proper secrets store in production)
"""

import os

import pandas as pd
import streamlit as st


def render() -> None:
    """Entry point called by app.py when the Staff Dashboard page is selected."""
    st.title("Internal Staff Dashboard")

    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        _render_login()
        return

    # Logout — top-right aligned
    _, col_logout = st.columns([5, 1])
    with col_logout:
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    _render_dashboard()


# ── Login ──────────────────────────────────────────────────────────────────────

def _render_login() -> None:
    col, _ = st.columns([1, 1])
    with col:
        st.markdown("### Staff Login")
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd  = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True, type="primary"):
                if user == "admin" and pwd == "admin":
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _render_dashboard() -> None:
    data_path = os.path.join("data", "simulation_data.csv")
    if not os.path.exists(data_path):
        st.info(
            "No simulation data found. Submit loan applications to populate this dashboard.",
            icon="ℹ️",
        )
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        return

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp", ascending=False)

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Applications", f"{len(df):,}")
    c2.metric("Total Loan Volume",  f"${df['Loan Amount'].sum():,.0f}")
    c3.metric("Avg Interest Rate",  f"{df['Expected Interest Rate'].mean():.2f}%")

    approval_rate = (df["Estimated Repay Score"] > 0.60).mean()
    c4.metric(
        "Approval Rate",
        f"{approval_rate:.1%}",
        help="Share of applications with a repay score above 60%",
    )

    # ── Recent applications table ──────────────────────────────────────────────
    st.markdown("### Recent Applications")
    st.dataframe(
        df.head(50),
        use_container_width=True,
        column_config={
            "Estimated Repay Score": st.column_config.ProgressColumn(
                "Repay Score",
                format="%.0%",
                min_value=0,
                max_value=1,
            ),
            "Loan Amount": st.column_config.NumberColumn(
                "Loan Amount",
                format="$%d",
            ),
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="YYYY-MM-DD HH:mm",
            ),
        },
    )

    # ── Analytics ──────────────────────────────────────────────────────────────
    st.markdown("### Analytics")

    if "Timestamp" in df.columns:
        st.markdown("**Application Volume Over Time**")
        df["Date"] = df["Timestamp"].dt.date
        daily = df.groupby("Date").size().rename("Applications")
        st.line_chart(daily)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Applications by Purpose**")
        st.bar_chart(df["Purpose"].value_counts())

    with col2:
        st.markdown("**Avg Repay Score by Purpose**")
        st.bar_chart(df.groupby("Purpose")["Estimated Repay Score"].mean())
