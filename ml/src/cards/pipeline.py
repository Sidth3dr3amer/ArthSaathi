"""
Credit Card Intelligence -> the full Tier 1-6 pipeline.

    User Profile -> Spending Analyzer -> Financial Twin      (Tier 1)
                 -> Reward | Lounge | Membership | Cost      (Tier 2)
                 -> 12-Month Simulation                      (Tier 3)
                 -> Multi-Agent Deliberation                 (Tier 4)
                 -> Ranking Engine                           (Tier 5)
                 -> Explanation Agent                        (Tier 6)

This is the "Final Architecture" section of the design doc below the Tier-0
line. Tier 0 (the daily knowledge-base cron) is upstream and offline; this
pipeline consumes the card knowledge base it produces.
"""

from __future__ import annotations

from typing import Any

from ..common import llm
from ..councils.growth.credit_card import (
    analyze_spend_profile,
    filter_eligible_cards,
    load_card_database,
    profile_to_engine_dict,
)
from ..schemas.profile import UserProfile
from ..schemas.state import FinancialState
from .tier1_profiler import run_tier1
from .tier2_evaluation import evaluate_all
from .tier3_twin import simulate_all
from .tier4_experts import deliberate
from .tier5_ranking import rank_cards
from .tier6_explain import build_explanation, explain


def run_card_intelligence(
    profile: UserProfile,
    cards: list[dict[str, Any]] | None = None,
    transactions: list[dict[str, Any]] | None = None,
    top_n: int = 5,
    use_llm: bool = False,
    with_arguments: bool = False,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Run all six tiers.

    Deterministic by default (`use_llm=False`), so the ranking is reproducible;
    the LLM only ever phrases explanations and expert arguments.
    """
    database = cards if cards is not None else load_card_database()
    if not database:
        return {
            "status": "no card database",
            "reason": "card knowledge base is empty",
            "tiers": {},
            "recommendation": None,
        }

    # ---- Tier 1 ---------------------------------------------------------
    tier1 = run_tier1(profile, transactions)
    engine_profile = profile_to_engine_dict(profile)
    spend_analysis = analyze_spend_profile(engine_profile)

    eligible, rejected = filter_eligible_cards(engine_profile, database)
    if not eligible:
        return {
            "status": "no eligible cards",
            "reason": (
                "every card was filtered out by eligibility -- most commonly a "
                "max_annual_fee of 0 against a database of fee-charging cards"
            ),
            "tiers": {"tier1": tier1},
            "cards_considered": len(database),
            "rejected": len(rejected),
            "recommendation": None,
        }

    # ---- Tiers 2 and 3 --------------------------------------------------
    evaluations = evaluate_all(eligible, engine_profile, spend_analysis, tier1["twin"])
    simulations = simulate_all(evaluations, tier1["twin"])

    # ---- Tier 4 ---------------------------------------------------------
    deliberation = deliberate(
        evaluations, tier1["twin"],
        with_arguments=with_arguments and use_llm, provider=provider,
    )

    # ---- Tier 5 ---------------------------------------------------------
    ranking = rank_cards(
        evaluations, simulations, deliberation, eligible,
        tier1["profiler"], tier1["spending"], tier1["twin"], top_n=top_n,
    )

    # ---- Tier 6 ---------------------------------------------------------
    explanation = explain(
        build_explanation(ranking, evaluations, simulations, deliberation),
        provider=provider, use_llm=use_llm,
    )

    return {
        "status": "complete",
        "cards_considered": len(database),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "tiers": {
            "tier1": tier1,
            "tier2": evaluations,
            "tier3": simulations,
            "tier4": deliberation,
            "tier5": ranking,
            "tier6": explanation,
        },
        "recommendation": {
            "card_name": ranking.get("winner"),
            "final_score_percent": (
                ranking["ranked"][0]["final_score_percent"] if ranking.get("ranked") else None
            ),
            "net_annual_value": (
                ranking["ranked"][0]["net_annual_value"] if ranking.get("ranked") else None
            ),
            "explanation": explanation.get("prose") or explanation.get("text", ""),
            "points": explanation.get("points", []),
        },
        "top_cards": ranking.get("ranked", []),
    }


def card_intelligence_node(
    state: FinancialState,
    cards: list[dict[str, Any]] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Adapter. Writes one result key."""
    return {
        "card_intelligence_result": run_card_intelligence(
            state["profile"],
            cards=cards,
            transactions=state.get("transactions"),
            top_n=top_n,
        )
    }
