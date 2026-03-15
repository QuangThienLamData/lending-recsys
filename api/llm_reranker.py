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

def _build_prompt(user_profile: dict, candidates: pd.DataFrame) -> str:
    """
    Construct a structured prompt for the LLM.

    Parameters
    ----------
    user_profile : dict   keys: annual_inc, dti, fico_range_low, fico_range_high,
                                home_ownership, addr_state
    candidates   : DataFrame  one row per candidate item with columns:
                                item_idx, item_id, grade, purpose, term,
                                int_rate, loan_amnt, positive_rate
    """
    profile_str = (
        f"- Annual income: ${user_profile.get('annual_inc', 'unknown'):,}\n"
        f"- Debt-to-income ratio: {user_profile.get('dti', 'unknown')}%\n"
        f"- FICO score: {user_profile.get('fico_range_low', '?')}–"
        f"{user_profile.get('fico_range_high', '?')}\n"
        f"- Home ownership: {user_profile.get('home_ownership', 'unknown')}\n"
        f"- State: {user_profile.get('addr_state', 'unknown')}"
    )

    items_str = "\n".join(
        f"  {i+1}. [{row['item_id']}] Grade={row['grade']}, "
        f"Purpose={row['purpose']}, Term={row['term']}, "
        f"Rate={row['int_rate']:.1f}%, AvgAmount=${row['loan_amnt']:,.0f}, "
        f"HistRepayRate={row['positive_rate']:.0%}"
        for i, (_, row) in enumerate(candidates.iterrows())
    )

    return (
        "You are a financial advisor helping rank loan products for a borrower.\n\n"
        "Borrower profile:\n"
        f"{profile_str}\n\n"
        "Candidate loan products (already pre-ranked by a machine learning model):\n"
        f"{items_str}\n\n"
        "Task: Reorder the products so the most suitable ones for this specific "
        "borrower appear first. Consider the borrower's income, credit score, and "
        "risk tolerance. Return ONLY a JSON array of the product item_ids in your "
        "preferred order, e.g.:\n"
        '["B_debt_consolidation_36 months", "A_home_improvement_60 months", ...]\n\n'
        "Your reranked order:"
    )


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
) -> pd.DataFrame:
    """
    Rerank candidate loan products using an LLM.

    Parameters
    ----------
    user_profile : dict    borrower attributes
    candidates   : pd.DataFrame  columns include item_id, grade, purpose, …
                                 already sorted by ranking model score

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
    prompt = _build_prompt(user_profile, candidates)

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
