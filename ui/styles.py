"""
ui/styles.py
Global CSS injected once at app startup.
Edit this file to change colors, spacing, typography, or component styling.
"""

import streamlit as st

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY   = "#2563EB"
SUCCESS   = "#059669"
DANGER    = "#DC2626"
AMBER     = "#D97706"
MUTED     = "#6B7280"
CARD_BG   = "rgba(255, 255, 255, 0.04)"
BORDER    = "rgba(255, 255, 255, 0.09)"


def inject_styles() -> None:
    """Inject global CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Metric cards ──────────────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background  : {CARD_BG};
            border      : 1px solid {BORDER};
            border-radius: 10px;
            padding     : 14px 18px;
        }}
        [data-testid="stMetricLabel"] > div {{
            font-size      : 0.75rem;
            color          : #9CA3AF;
            text-transform : uppercase;
            letter-spacing : 0.06em;
            font-weight    : 600;
        }}
        [data-testid="stMetricValue"] > div {{
            font-size  : 1.25rem;
            font-weight: 700;
        }}

        /* ── Buttons ───────────────────────────────────────────────────────── */
        button[kind="primary"] {{
            background   : {PRIMARY};
            border-radius: 8px;
            font-weight  : 600;
            font-size    : 1rem;
            letter-spacing: 0.01em;
            transition   : background 0.18s, box-shadow 0.18s;
        }}
        button[kind="primary"]:hover {{
            background  : #1D4ED8;
            box-shadow  : 0 4px 16px rgba(37,99,235,0.35);
        }}
        button[kind="secondary"] {{
            border-radius: 8px;
            font-weight  : 500;
        }}

        /* ── Section label helper (used via unsafe_allow_html) ─────────────── */
        .ui-section-label {{
            font-size      : 0.72rem;
            font-weight    : 700;
            text-transform : uppercase;
            letter-spacing : 0.09em;
            color          : #6B7280;
            margin         : 22px 0 6px 0;
        }}

        /* ── Progress bar ──────────────────────────────────────────────────── */
        .stProgress > div > div > div {{
            border-radius: 8px;
        }}

        /* ── Alert boxes ───────────────────────────────────────────────────── */
        [data-testid="stAlert"] {{
            border-radius: 8px;
        }}

        /* ── Expander ──────────────────────────────────────────────────────── */
        [data-testid="stExpander"] > details > summary {{
            font-weight: 500;
        }}

        /* ── Divider ───────────────────────────────────────────────────────── */
        hr {{
            border-color: {BORDER} !important;
            margin      : 1.5rem 0 !important;
        }}

        /* ── Sidebar ───────────────────────────────────────────────────────── */
        [data-testid="stSidebarContent"] {{
            padding-top: 1rem;
        }}

        /* ── Data table ────────────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow     : hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
