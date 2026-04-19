"""
api/llm_reranker.py
--------------------
Optional LLM-based reranker.  Uses an LLM (OpenAI-compatible API or Anthropic
Claude) to reorder a short list of candidate loan products based on a natural
language description of the borrower's profile.

Environment variables:
  LLM_PROVIDER   = "openai" | "anthropic"  (default "openai")
  OPENAI_API_KEY = sk-…
  ANTHROPIC_API_KEY = sk-ant-…

If no API key is set the function returns the candidates unchanged and logs
a warning — so the rest of the pipeline still works without the LLM.
"""

import os
import json
import re
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MAX_TOKENS   = 512


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(user_profile: dict, candidates: pd.DataFrame,
                  loan_request: dict, user_prompt: str = None) -> str:
    """
    Construct a structured prompt for the LLM.

    Parameters
    ----------
    user_profile : dict   keys: annual_inc, dti, fico_range_low, fico_range_high,
                                home_ownership, addr_state
    candidates   : DataFrame  one row per candidate item with columns:
                                item_idx, item_id, grade, purpose, term,
                                int_rate, loan_amnt, positive_rate
    loan_request : dict   keys: purpose, loan_amnt, int_rate
    """
    inc = user_profile.get('annual_inc')
    inc_str = f"${inc:,.0f}" if isinstance(inc, (int, float)) else "unknown"
    profile_str = (
        f"- Annual income: {inc_str}\n"
        f"- Debt-to-income ratio: {user_profile.get('dti', 'unknown')}%\n"
        f"- FICO score: {user_profile.get('fico_range_low', '?')}–"
        f"{user_profile.get('fico_range_high', '?')}\n"
        f"- Home ownership: {user_profile.get('home_ownership', 'unknown')}\n"
        f"- State: {user_profile.get('addr_state', 'unknown')}"
    )

    purpose   = loan_request.get("purpose") or "any"
    loan_amnt = loan_request.get("loan_amnt")
    int_rate  = loan_request.get("int_rate")

    request_lines = [f"- Requested purpose: {purpose.replace('_', ' ')}"]
    if loan_amnt:
        request_lines.append(f"- Requested loan amount: ${loan_amnt:,.0f}")
    if int_rate:
        request_lines.append(f"- Requested interest rate: {int_rate:.1f}%")
    request_str = "\n".join(request_lines)

    items_str = "\n".join(
        f"  {i+1}. [{row['item_id']}] Grade={row['grade']}, "
        f"Purpose={row['purpose']}, Term={row['term']}, "
        f"Rate={row['int_rate']:.1f}%, AvgAmount=${row['loan_amnt']:,.0f}, "
        f"HistRepayRate={row['positive_rate']:.0%}"
        for i, (_, row) in enumerate(candidates.iterrows())
    )

    instructions = (
        "You are a loan product ranker. Your job is to reorder loan products strictly "
        "according to how well they match the borrower's explicit request — "
        "IGNORE the machine learning model's original order entirely.\n\n"
        "Borrower's loan request (these are the PRIMARY criteria):\n"
        f"{request_str}\n\n"
        "Borrower profile (secondary context):\n"
        f"{profile_str}\n\n"
        "Candidate loan products:\n"
        f"{items_str}\n\n"
    )

    if user_prompt and user_prompt.strip():
        instructions += (
            f"USER'S SPECIFIC SCENARIO AND PROMPT: {user_prompt.strip()}\n\n"
            "STRICT ranking rules based on the user's prompt above — apply exactly in this order:\n"
            "  1. PURPOSE DEDUCTION: Infer the actual semantic purpose from the user's prompt (e.g., buying a car implies Purpose='car', renovating a house implies Purpose='home_improvement'). You MUST absolutely prioritize products matching this inferred purpose at the very top. Ignore the requested purpose in the form if it contradicts the prompt.\n"
            "  2. LOAN AMOUNT MATH: Calculate the exact amount they need (e.g., total asset price minus cash they have). Rank the products that have an AvgAmount as close as possible to this required amount.\n"
            "  3. FINANCIAL BENEFITS: Among items with the correct purpose and similar required amounts, strictly sort by:\n"
            "      a. Lowest Rate (Interest Rate) first.\n"
            "      b. Shortest Term first (e.g., '36 months' preferred over '60 months').\n"
            "      c. Highest Grade (Grade A is best, then B, etc.).\n"
        )
    else:
        instructions += (
            "STRICT ranking rules — apply in order:\n"
            "  1. PURPOSE (most important): Products whose Purpose exactly matches the "
            f"requested purpose '{purpose}' MUST come before any non-matching purpose. "
            "This rule overrides all other criteria.\n"
            "  2. LOAN AMOUNT: Among products with the same purpose priority, rank by "
            "closest AvgAmount to the requested amount.\n"
            "  3. INTEREST RATE: Among ties, rank by closest Rate to the requested rate.\n"
            "  4. BORROWER FIT: Use FICO, income, DTI as a tiebreaker only.\n"
        )

    instructions += (
        "\nReturn ONLY a JSON array of ALL item_ids in your reranked order. "
        "Do not drop any items. Example:\n"
        '["B_car_36 months", "A_car_60 months", "C_debt_consolidation_36 months"]\n\n'
        "Reranked order:"
    )
    return instructions


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp   = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp   = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _parse_response(response_text: str, original_item_ids: List[str]) -> List[str]:
    """Extract the reranked item_id list from the LLM response."""
    # Try to find a JSON array in the response
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if match:
        try:
            reranked = json.loads(match.group())
            # Validate that returned IDs are a subset of original
            valid = [iid for iid in reranked if iid in set(original_item_ids)]
            # Append any IDs the LLM missed at the end
            missing = [iid for iid in original_item_ids if iid not in valid]
            return valid + missing
        except json.JSONDecodeError:
            pass
    logger.warning("LLM response could not be parsed; returning original order.")
    return original_item_ids


# ── Public API ────────────────────────────────────────────────────────────────

def llm_rerank(
    user_profile: dict,
    candidates: pd.DataFrame,
    loan_request: dict = None,
    user_prompt: str = None,
) -> pd.DataFrame:
    """
    Rerank candidate loan products using an LLM.

    Parameters
    ----------
    user_profile : dict    borrower demographics
    candidates   : pd.DataFrame  columns include item_id, grade, purpose, …
                                 already sorted by ranking model score
    loan_request : dict    user's loan request: purpose, loan_amnt, int_rate

    Returns
    -------
    pd.DataFrame reordered by LLM preference
    """
    # Check for API keys
    provider = LLM_PROVIDER.lower()
    has_key  = (
        (provider == "openai"    and "OPENAI_API_KEY"    in os.environ) or
        (provider == "anthropic" and "ANTHROPIC_API_KEY" in os.environ)
    )

    if not has_key:
        logger.warning(
            "LLM reranker: no API key found for provider '%s'. "
            "Skipping LLM rerank and returning ranking model order.", provider
        )
        return candidates

    original_ids = candidates["item_id"].tolist()
    prompt = _build_prompt(user_profile, candidates, loan_request or {}, user_prompt)

    try:
        if provider == "openai":
            raw_response = _call_openai(prompt)
        else:
            raw_response = _call_anthropic(prompt)

        reranked_ids = _parse_response(raw_response, original_ids)

        # Reorder DataFrame
        id_to_row = {row["item_id"]: row for _, row in candidates.iterrows()}
        reranked_rows = [id_to_row[iid] for iid in reranked_ids if iid in id_to_row]
        return pd.DataFrame(reranked_rows).reset_index(drop=True)

    except Exception as exc:
        logger.error("LLM reranker failed: %s. Returning ranking model order.", exc)
        return candidates
