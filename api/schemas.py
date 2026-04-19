"""
api/schemas.py
---------------
Pydantic v2 request / response models for the FastAPI endpoints.
These power both input validation and the auto-generated OpenAPI docs.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="External member_id from LendingClub dataset",
        examples=["1234567"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of loan products to return (1-100)",
    )
    use_llm_rerank: bool = Field(
        default=False,
        description="If True, apply LLM-based reranking as the final step",
    )
    llm_pool: int = Field(
        default=20,
        ge=1,
        le=100,
        description="How many top candidates from the ranking model to pass to the LLM reranker",
    )
    retrieval_pool: int = Field(
        default=100,
        ge=10,
        le=500,
        description="How many FAISS candidates to retrieve before ranking",
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Optional natural language prompt outlining user's specific context and goals",
    )

    # ── Optional demographics for cold-start XGBoost scoring ─────────────
    annual_inc:     Optional[float] = Field(default=None, description="Annual income (USD)")
    dti:            Optional[float] = Field(default=None, description="Debt-to-income ratio")
    fico_range_low: Optional[float] = Field(default=None, description="FICO score lower bound")
    fico_range_high:Optional[float] = Field(default=None, description="FICO score upper bound")
    home_ownership: Optional[str]   = Field(default=None, description="RENT | OWN | MORTGAGE | OTHER")
    addr_state:     Optional[str]   = Field(default=None, description="Two-letter US state code (e.g. CA)")

    # ── Loan request parameters (used for approval XGBoost scoring) ───────
    loan_amnt_request: Optional[float] = Field(default=None, description="Requested loan amount (USD)")
    int_rate_request:  Optional[float] = Field(default=None, description="Expected interest rate (%)")
    purpose_request:   Optional[str]   = Field(default=None, description="Loan purpose (e.g. debt_consolidation)")
    term_request:      Optional[str]   = Field(default=None, description="Loan term: '36 months' or '60 months'")


# ── Response ─────────────────────────────────────────────────────────────────

class ItemDetail(BaseModel):
    item_idx: int        = Field(..., description="Internal integer item index")
    item_id:  str        = Field(..., description="Synthetic item ID: grade_purpose_term")
    grade:    str        = Field(..., description="LendingClub loan grade (A–G)")
    purpose:  str        = Field(..., description="Loan purpose category")
    term:     str        = Field(..., description="Loan term (36 months / 60 months)")
    int_rate: float      = Field(..., description="Average interest rate for this product")
    loan_amnt: float     = Field(..., description="Average loan amount for this product")
    positive_rate:  float = Field(..., description="Historical repayment rate (0–1)")
    xgb_repay_prob: float = Field(default=0.0, description="XGBoost predicted repayment probability (0–1)")
    rank:     int        = Field(..., description="Rank in the recommendation list (1-based)")
    score:    float      = Field(..., description="Relevance score from ranking model")


class RecommendResponse(BaseModel):
    user_id:         str             = Field(..., description="Echo of the request user_id")
    n_returned:      int             = Field(..., description="Number of items in the response")
    pipeline_stages: List[str]       = Field(..., description="Stages applied: retrieval, ranking, llm")
    recommendations: List[ItemDetail]= Field(..., description="Ranked loan product recommendations")
    approved:        bool            = Field(default=False, description="Whether the loan request is approved")
    approval_score:  float           = Field(default=0.0,   description="XGBoost approval probability (0–1)")
    shap_features:   List[dict]      = Field(default_factory=list, description="SHAP values per user feature")
    improvements:    List[dict]      = Field(default_factory=list, description="Minimum changes to get approved")
    user_profile:    dict            = Field(default_factory=dict, description="Raw user demographic values used for scoring")
    llm_advice:      str             = Field(default="", description="LLM-generated personalised financial advice")


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str       = "ok"
    artefacts_loaded: List[str] = Field(default_factory=list)
