"""
api/explain.py
--------------
SHAP-based explanation for the XGBoost repayment approval model.

Given a user's feature vector and item feature vector, produces:
  1. Per-feature SHAP values (user features only, grouped by original field)
  2. Minimum improvement suggestions — the smallest change to each numeric
     user field that would push the approval score above the threshold.
"""

import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD = 0.65

# Human-readable labels for user features
FEATURE_LABELS = {
    "annual_inc":      "Annual Income ($)",
    "dti":             "Debt-to-Income Ratio",
    "fico_range_low":  "FICO Score (Low)",
    "fico_range_high": "FICO Score (High)",
    "home_ownership":  "Home Ownership",
    "addr_state":      "State",
    "loan_amnt":       "Requested Loan Amount ($)",
}

# For numeric features: direction that improves repayment probability
# +1 = higher is better,  -1 = lower is better
IMPROVE_DIRECTION = {
    "annual_inc":      +1,
    "dti":             -1,   # lower DTI = less debt burden
    "fico_range_low":  +1,
    "fico_range_high": +1,
    "loan_amnt":       -1,   # lower amount requested = less debt burden
}

# Bounds for binary search (raw / unscaled values)
SEARCH_BOUNDS = {
    "annual_inc":      (0,       500_000),
    "dti":             (0,       50),
    "fico_range_low":  (300,     850),
    "fico_range_high": (300,     850),
    "loan_amnt":       (500,     40_000),
}


# ── Helper: predict score from raw user and item dict ────────────────────────

def _score_from_raw_both(raw_profile: dict, raw_loan_request: dict,
                         predictor, artefacts: dict) -> float:
    """Transform a raw user dict and active loan request → feature vec → XGBoost score."""
    # 1. User Feature
    user_transformer = artefacts.get("user_transformer")
    top_states  = artefacts.get("top_states", [])

    state = raw_profile.get("addr_state", "other")
    if top_states and state not in top_states:
        state = "other"

    u_row = {
        "annual_inc":      raw_profile.get("annual_inc"),
        "dti":             raw_profile.get("dti"),
        "fico_range_low":  raw_profile.get("fico_range_low"),
        "fico_range_high": raw_profile.get("fico_range_high"),
        "home_ownership":  raw_profile.get("home_ownership", "Unknown"),
        "addr_state":      state,
    }
    user_feat = user_transformer.transform(pd.DataFrame([u_row])).astype(np.float32)[0]

    # 2. Item Feature
    item_transformer = artefacts.get("item_transformer")
    i_row = {
        "int_rate":  raw_loan_request.get("int_rate")  if raw_loan_request.get("int_rate")  is not None else 12.0,
        "loan_amnt": raw_loan_request.get("loan_amnt"),
        "grade":     "Unknown",
        "purpose":   raw_loan_request.get("purpose")   if raw_loan_request.get("purpose")   is not None else "other",
        "term":      raw_loan_request.get("term")      if raw_loan_request.get("term")      is not None else "Unknown",
    }
    item_feat = item_transformer.transform(pd.DataFrame([i_row])).astype(np.float32)[0]

    import xgboost as xgb
    X = np.hstack([user_feat.reshape(1, -1), item_feat.reshape(1, -1)])
    prob = predictor.repay_predictor.model.predict(xgb.DMatrix(X)).astype(np.float32)
    if getattr(predictor.repay_predictor, "calibrator", None) is not None:
        return float(predictor.repay_predictor.calibrator.predict(prob)[0])
    return float(prob[0])


# ── SHAP values ───────────────────────────────────────────────────────────────

def compute_shap_values(user_feat_vec: np.ndarray, item_feat_vec: np.ndarray,
                        predictor, artefacts: dict) -> list[dict]:
    """
    Compute SHAP values for the user features in the approval prediction.
    Uses XGBoost's built-in pred_contribs (no external shap package needed).

    Returns a list of dicts (one per original user field):
      {field, label, shap_value}
    Sorted by |shap_value| descending.
    """
    import xgboost as xgb

    transformer = artefacts.get("user_transformer")
    if transformer is None or predictor.repay_predictor is None:
        return []

    # Build combined feature vector
    X = np.hstack([user_feat_vec.reshape(1, -1),
                   item_feat_vec.reshape(1, -1)]).astype(np.float32)

    # pred_contribs=True returns SHAP values (last col is bias — drop it)
    contribs    = predictor.repay_predictor.model.predict(
        xgb.DMatrix(X), pred_contribs=True
    )
    shap_values = contribs[0, :-1]   # shape (total_feat_dim,)

    user_feat_dim = len(user_feat_vec)
    user_shap     = shap_values[:user_feat_dim]  # only user features

    # Get feature names from transformer
    feat_names = transformer.get_feature_names_out().tolist()  # length = user_feat_dim

    # Group OHE columns back to their original field
    grouped = {}   # field_name -> cumulative shap value
    for fname, sv in zip(feat_names, user_shap):
        # fname like "num__annual_inc" or "cat__home_ownership_MORTGAGE"
        if fname.startswith("num__"):
            field = fname[len("num__"):]
        elif fname.startswith("cat__"):
            # "cat__home_ownership_MORTGAGE" -> "home_ownership"
            rest  = fname[len("cat__"):]
            field = "_".join(rest.split("_")[:-1]) if "_" in rest else rest
            # Handle multi-word fields like addr_state
            # Try to match known fields
            matched = next(
                (f for f in FEATURE_LABELS if rest.startswith(f + "_") or rest == f),
                None
            )
            if matched:
                field = matched
        else:
            field = fname

        grouped[field] = grouped.get(field, 0.0) + float(sv)

    result = []
    for field, sv in grouped.items():
        if field not in FEATURE_LABELS:
            continue
        result.append({
            "field":     field,
            "label":     FEATURE_LABELS[field],
            "shap_value": round(sv, 4),
        })

    return sorted(result, key=lambda x: abs(x["shap_value"]), reverse=True)


# ── Minimum improvement suggestions ──────────────────────────────────────────

def find_improvements(raw_profile: dict, raw_loan_request: dict,
                      current_score: float, predictor, artefacts: dict,
                      threshold: float = APPROVAL_THRESHOLD) -> list[dict]:
    """
    For each numeric user and loan field, find the minimum value change that would push
    the approval score above threshold (holding all other fields fixed).

    Returns a list of dicts:
      {field, label, current_value, min_required_value, direction, achievable}
    Only includes fields where the current value is below what's needed.
    """
    if predictor.repay_predictor is None:
        return []

    suggestions = []

    for field, direction in IMPROVE_DIRECTION.items():
        is_user = field in raw_profile
        is_item = field in raw_loan_request
        
        current_val = raw_profile.get(field) if is_user else raw_loan_request.get(field)
        if current_val is None:
            continue

        lo, hi = SEARCH_BOUNDS[field]

        # Check if even the best possible value would be enough
        best_val     = hi if direction == +1 else lo
        best_profile = {**raw_profile}
        best_item    = {**raw_loan_request}
        if is_user: best_profile[field] = best_val
        else: best_item[field] = best_val
        
        best_score   = _score_from_raw_both(best_profile, best_item, predictor, artefacts)

        if best_score < threshold:
            # Not achievable by changing this field alone
            suggestions.append({
                "field":            field,
                "label":            FEATURE_LABELS[field],
                "current_value":    round(float(current_val), 2),
                "min_required":     None,
                "achievable":       False,
            })
            continue

        if current_score >= threshold:
            # Already approved — show current value, no change needed
            suggestions.append({
                "field":            field,
                "label":            FEATURE_LABELS[field],
                "current_value":    round(float(current_val), 2),
                "min_required":     round(float(current_val), 2),
                "achievable":       True,
            })
            continue

        # Binary search for the minimal value that crosses the threshold
        a, b = (current_val, hi) if direction == +1 else (lo, current_val)
        for _ in range(40):   # 40 iterations → ~1e-12 relative precision
            mid = (a + b) / 2
            trial_profile = {**raw_profile}
            trial_item    = {**raw_loan_request}
            if is_user: trial_profile[field] = mid
            else: trial_item[field] = mid
            
            score = _score_from_raw_both(trial_profile, trial_item, predictor, artefacts)
            if score >= threshold:
                b = mid
            else:
                a = mid

        min_required = b if direction == +1 else a

        # Only suggest if actual improvement is needed
        if (direction == +1 and min_required > current_val + 0.5) or \
           (direction == -1 and min_required < current_val - 0.5):
            suggestions.append({
                "field":            field,
                "label":            FEATURE_LABELS[field],
                "current_value":    round(float(current_val), 2),
                "min_required":     round(float(min_required), 2),
                "achievable":       True,
            })

    return sorted(suggestions, key=lambda x: x.get("min_required") is None)


# ── Home ownership suggestion ─────────────────────────────────────────────────

def find_best_home_ownership(raw_profile: dict, raw_loan_request: dict,
                             predictor, artefacts: dict) -> dict | None:
    """
    Try all home_ownership categories and return the one with the highest score.
    Returns None if the current category is already the best.
    """
    if predictor.repay_predictor is None:
        return None

    categories   = ["OWN", "MORTGAGE", "RENT", "OTHER"]
    current_cat  = raw_profile.get("home_ownership", "RENT")
    best_cat, best_score = current_cat, -1.0

    for cat in categories:
        trial = {**raw_profile, "home_ownership": cat}
        s     = _score_from_raw_both(trial, raw_loan_request, predictor, artefacts)
        if s > best_score:
            best_score, best_cat = s, cat

    if best_cat != current_cat:
        return {
            "field":            "home_ownership",
            "label":            FEATURE_LABELS["home_ownership"],
            "current_value":    current_cat,
            "recommended":      best_cat,
            "score_with_recommended": round(best_score, 4),
        }
    return None


# ── LLM financial advice ──────────────────────────────────────────────────────

def generate_llm_advice(
    user_profile:    dict,
    shap_features:   list,
    improvements:    list,
    approval_score:  float,
    approved:        bool,
    loan_request:    dict,
    threshold:       float = APPROVAL_THRESHOLD,
) -> str:
    """
    Call OpenAI (or Anthropic) to generate personalised financial advice
    based on the SHAP explanation and improvement suggestions.
    Returns an advice string, or empty string if no API key is available.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    has_key  = (
        (provider == "openai"    and os.getenv("OPENAI_API_KEY"))    or
        (provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
    )
    if not has_key:
        logger.warning("LLM advice: no API key — skipping")
        return ""

    # ── Build prompt ──────────────────────────────────────────────────────────
    status_str = "APPROVED" if approved else "NOT APPROVED"
    score_str  = f"{approval_score:.1%}"

    profile_lines = "\n".join([
        f"  - Annual Income:    ${user_profile.get('annual_inc', 0):,.0f}",
        f"  - DTI Ratio:        {user_profile.get('dti', '?')}%",
        f"  - FICO Score:       {user_profile.get('fico_range_low','?')} – {user_profile.get('fico_range_high','?')}",
        f"  - Home Ownership:   {user_profile.get('home_ownership','?')}",
        f"  - State:            {user_profile.get('addr_state','?')}",
    ])

    shap_lines = "\n".join([
        f"  - {f['label']}: {'+' if f['shap_value'] >= 0 else ''}{f['shap_value']:.3f} "
        f"({'positively' if f['shap_value'] >= 0 else 'negatively'} impacts repayment probability)"
        for f in shap_features
    ])

    imp_lines = ""
    if improvements and not approved:
        actionable = [i for i in improvements if i.get("min_required") is not None
                      and i["min_required"] != i["current_value"]]
        if actionable:
            imp_lines = "\nMinimum changes to reach approval threshold:\n" + "\n".join([
                f"  - {i['label']}: {i['current_value']} → {i['min_required']}"
                for i in actionable
            ])

    loan_lines = (
        f"  - Purpose: {loan_request.get('purpose','?').replace('_',' ')}\n"
        f"  - Amount:  ${loan_request.get('loan_amnt', 0):,.0f}\n"
        f"  - Term:    {loan_request.get('term','?')}\n"
        f"  - Rate:    {loan_request.get('int_rate','?')}%"
    )

    prompt = (
        f"You are a financial advisor. A loan applicant has received a repayment "
        f"score of {score_str} and their application is {status_str} "
        f"(threshold: {threshold:.0%}).\n\n"
        f"Loan request:\n{loan_lines}\n\n"
        f"Applicant financial profile:\n{profile_lines}\n\n"
        f"SHAP feature contributions to their repayment score:\n{shap_lines}\n"
        f"{imp_lines}\n\n"
        f"Write 3–5 sentences of concise, actionable financial advice for this "
        f"applicant. Focus on:\n"
        f"1. The most impactful factors (highest |SHAP value|) driving their score.\n"
        f"2. Specific steps they can take to improve their financial profile.\n"
        f"3. If not approved, the most achievable path to approval.\n"
        f"Be encouraging but realistic. Do not repeat the numbers verbatim — "
        f"explain them in plain language."
    )

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp   = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
    except Exception as exc:
        logger.error("LLM advice failed: %s", exc)
        return ""
