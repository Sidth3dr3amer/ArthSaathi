"""
Credit-card endpoints.

These sit on `councils.growth.credit_card`, which is the working recommendation
engine. The Tier 1-6 system in `ml/src/cards/` is being built separately; when
it lands it can be exposed here additively without changing these routes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ml.src.councils.growth.credit_card import (
    analyze_spend_profile,
    credit_card_node,
    load_card_database,
    profile_to_engine_dict,
)
from ml.src.schemas.state import new_state

from ..deps import resolve_profile
from ..schemas import WorkflowRunRequest

router = APIRouter(prefix="/cards", tags=["cards"])


@router.get("")
def list_cards() -> dict[str, Any]:
    """The curated card database the recommendation engine scores against."""
    cards = load_card_database()
    return {
        "count": len(cards),
        "cards": [
            {
                "card_name": c.get("card_name"),
                "issuer": c.get("issuer"),
                "card_type": c.get("card_type"),
                "annual_fee": c.get("annual_fee"),
                "base_reward_rate": c.get("base_reward_rate"),
            }
            for c in cards
        ],
    }


@router.post("/recommend")
def recommend(request: WorkflowRunRequest, top_n: int = Query(default=3, ge=1, le=10)) -> dict[str, Any]:
    """Rank cards for a user."""
    profile, source = resolve_profile(request.user_id, request.profile)

    try:
        result = credit_card_node(
            new_state(profile, request.query), top_n=top_n
        )["credit_card_result"]
    except Exception as exc:
        return {"recommendations": [], "errors": [repr(exc)], "profile_source": source}

    return {
        "cards_considered": result["cards_considered"],
        "eligible_count": result.get("eligible_count", 0),
        "rejected_count": result.get("rejected_count", 0),
        "spend_analysis": result.get("spend_analysis"),
        "recommendations": [
            {
                "card_name": r["card"].get("card_name"),
                "issuer": r["card"].get("issuer"),
                "net_annual_value": r["net_value"],
                "annual_fee": r["card"].get("annual_fee"),
                "valuation": r["valuation"],
            }
            for r in result.get("recommendations", [])
        ],
        "routing": result.get("routing", {}),
        "profile_source": source,
        "errors": [],
    }


@router.post("/spend-profile")
def spend_profile(request: WorkflowRunRequest) -> dict[str, Any]:
    """The spend analysis behind a recommendation, for the dashboard breakdown."""
    profile, _ = resolve_profile(request.user_id, request.profile)
    try:
        return analyze_spend_profile(profile_to_engine_dict(profile))
    except Exception as exc:
        return {"errors": [repr(exc)]}
