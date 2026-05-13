# UI Folder Structure

This document maps every file in `ui/` to its purpose so you know exactly
where to go when adjusting any part of the interface.

---

## Directory Map

```
ui/
├── UI_STRUCTURE.md              ← this file
├── __init__.py
│
├── styles.py                    ← Global CSS (colors, spacing, typography)
│
├── components/                  ← Reusable UI building blocks
│   ├── __init__.py
│   ├── approval_banner.py       ← APPROVED / NOT APPROVED decision banner
│   ├── loan_card.py             ← Single loan product card + GRADE_COLOR map
│   └── analysis_panel.py       ← Post-decision analysis components:
│                                    render_applicant_summary()
│                                    render_profile()
│                                    render_shap_chart()
│                                    render_improvements()
│                                    render_llm_advice()
│
└── pages/                       ← Full-page render functions
    ├── __init__.py
    ├── loan_application.py      ← Loan Application page (form + results)
    └── staff_dashboard.py       ← Internal Staff Dashboard (login + analytics)
```

---

## Quick Edit Guide

| What you want to change | File to edit |
|---|---|
| Global colors, button style, metric card style | `styles.py` |
| Grade color badges (A–G) | `components/loan_card.py` → `GRADE_COLOR` |
| Loan card layout (fields, metric order) | `components/loan_card.py` → `render_loan_card()` |
| APPROVED / NOT APPROVED banner copy or colors | `components/approval_banner.py` |
| SHAP bar chart appearance | `components/analysis_panel.py` → `render_shap_chart()` |
| Financial profile fields shown | `components/analysis_panel.py` → `render_profile()` |
| Improvement suggestion copy | `components/analysis_panel.py` → `render_improvements()` |
| AI advisor panel style | `components/analysis_panel.py` → `render_llm_advice()` |
| Applicant summary (5-metric row) | `components/analysis_panel.py` → `render_applicant_summary()` |
| Loan application form fields | `pages/loan_application.py` |
| Approval result ordering / section visibility | `pages/loan_application.py` |
| Staff dashboard KPIs | `pages/staff_dashboard.py` → `_render_dashboard()` |
| Staff login credentials | `pages/staff_dashboard.py` → `_render_login()` |
| Sidebar layout / navigation | `app.py` (sidebar block) |
| Page routing | `app.py` (route block at bottom) |

---

## Data Flow

```
app.py
  ├─ load_artefacts()         loads ML models once (cached)
  ├─ inject_styles()          injects global CSS from styles.py
  ├─ sidebar                  navigation + advanced settings → settings dict
  └─ routes to:
       ├─ pages/staff_dashboard.py   render()
       └─ pages/loan_application.py  render(artefacts, purposes, get_name_fn, settings)
              ├─ form widgets
              ├─ run_recommendation_pipeline()
              └─ components/
                   ├─ approval_banner.render_approval_banner()
                   ├─ loan_card.render_loan_card()   (per item)
                   └─ analysis_panel.*
```

---

## Design Tokens

Defined in `styles.py` at the top as module-level constants:

| Token | Value | Used for |
|---|---|---|
| `PRIMARY` | `#2563EB` | Buttons, links, AI advisor border |
| `SUCCESS` | `#059669` | Approved banner, positive SHAP, high repay bars |
| `DANGER`  | `#DC2626` | Not-approved banner, negative SHAP, low repay bars |
| `AMBER`   | `#D97706` | Medium repay probability |
| `MUTED`   | `#6B7280` | Secondary labels, empty states |
