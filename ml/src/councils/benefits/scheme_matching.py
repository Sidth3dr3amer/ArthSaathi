"""
Benefits Council -> Scheme Matching Agent.

New agent. Eligibility answers "can they get it"; matching answers "is it worth
their time". Those are different questions and conflating them produces the
familiar useless output: forty schemes listed alphabetically.

Match score = benefit_weight x need_weight x effort_weight

  benefit  annual rupee value, log-scaled so a Rs 1 crore credit line does not
           drown out a Rs 6,000 transfer the user will actually claim
  need     how much this user's situation calls for that category -- an
           uninsured user needs health cover; a farmer needs crop credit
  effort   penalised by document count, since paperwork is the real reason
           entitlements go unclaimed in practice

This is the engine behind the deck's "AI Match Score" and "Why this scheme?"
panels, so every score carries its reasons.
"""

from __future__ import annotations

import math
from typing import Any

from ...schemas.profile import UserProfile
from ...schemas.state import FinancialState
from .eligibility import eligibility_advisor

#: Baseline importance of each scheme category, before user-specific need.
CATEGORY_BASE = {
    "health": 1.0,
    "insurance": 0.95,
    "income_support": 0.9,
    "pension": 0.75,
    "housing": 0.8,
    "credit": 0.7,
    "employment": 0.85,
    "social_security": 0.8,
    "savings": 0.55,
    "banking": 0.5,
    "subsidy": 0.6,
}

#: Effort proxy: each required document reduces the score slightly.
EFFORT_PER_DOCUMENT = 0.04
MIN_EFFORT_WEIGHT = 0.6

#: What fraction of the headline amount a user actually realises in a year.
#:
#: This matters more than it looks. A naive reading values PMSBY -- a Rs 20/year
#: accident policy -- at its Rs 2,00,000 sum assured, which would rank it above
#: PM-KISAN's guaranteed Rs 6,000 transfer and produce an obviously wrong
#: recommendation. An insurance payout is contingent; a cash transfer is not.
#: For risk products the honest annual value is the risk-transfer benefit, which
#: is closer to the commercial premium than to the sum assured.
REALISATION = {
    "cash_transfer": 1.00,     # paid, guaranteed
    "wage_employment": 0.80,   # depends on days actually worked
    "subsidy": 0.90,
    "banking": 0.30,           # access value, not a payout
    "insurance": 0.03,         # contingent; approximates risk-transfer value
    "pension": 0.60,           # deferred, but certain
    "savings": 0.08,           # own principal -- value is the rate advantage
    "loan": 0.06,              # value is interest saved, not the principal
    "credit_line": 0.05,
    "interest_subsidy": 0.50,
}


def _annual_value(benefit: dict[str, Any]) -> float:
    """
    Indicative annual rupee value the user actually realises.

    Combines the payout frequency with a realisation factor for the benefit
    type, so contingent and guaranteed benefits are comparable.
    """
    amount = float(benefit.get("amount", 0) or 0)
    if amount <= 0:
        return 0.0

    frequency = benefit.get("frequency", "one_time")
    if frequency == "annual":
        annualised = amount
    elif frequency == "per_season":
        annualised = amount * 2          # two cropping seasons a year
    elif frequency == "revolving":
        annualised = amount
    elif frequency == "one_time":
        annualised = amount / 5          # amortised over five years
    else:
        annualised = amount

    realisation = REALISATION.get(benefit.get("type", ""), 0.5)
    return annualised * realisation


def _need_weight(scheme: dict[str, Any], profile: UserProfile) -> tuple[float, list[str]]:
    """How much this user's situation calls for this scheme. Returns (weight, reasons)."""
    category = scheme.get("category", "")
    weight = CATEGORY_BASE.get(category, 0.5)
    reasons: list[str] = []

    benefit_type = (scheme.get("benefit") or {}).get("type", "")
    # "insurance" spans life, accident and crop cover, which answer completely
    # different needs. Distinguish them so crop insurance is not recommended
    # because the user has dependants.
    is_crop_cover = "farmer" in (scheme.get("eligibility", {}).get("occupation") or [])

    if category == "health" and not profile.has_health_insurance:
        weight *= 1.6
        reasons.append("no existing health cover")

    if category == "insurance" and is_crop_cover:
        if profile.occupation == "farmer":
            weight *= 1.5
            reasons.append("farming income exposed to crop loss")
    elif category == "insurance" and not profile.has_life_insurance and profile.dependents > 0:
        weight *= 1.4
        reasons.append(f"{profile.dependents} dependants and no life cover")

    if category == "pension" and profile.age >= 35:
        weight *= 1.3
        reasons.append("retirement horizon shortening")

    if category == "credit" and profile.monthly_surplus < 0:
        weight *= 1.3
        reasons.append("cash flow is negative")

    if category == "income_support" and profile.monthly_income < 25_000:
        weight *= 1.5
        reasons.append("low recorded income")

    if category == "employment" and profile.monthly_income < 15_000:
        weight *= 1.5
        reasons.append("income below a subsistence threshold")

    if category == "savings" and profile.monthly_surplus <= 0:
        weight *= 0.4
        reasons.append("no surplus available to save")

    if category == "housing" and profile.residence:
        weight *= 1.2
        reasons.append(f"{profile.residence} housing scheme applies")

    # A scheme whose rules name this user's occupation is aimed squarely at
    # them, and a guaranteed transfer beats a contingent one. Without this,
    # PM-KISAN -- the canonical recommendation for a smallholder -- ranks below
    # generic products purely because its headline number is small.
    targeted_occupations = (scheme.get("eligibility", {}) or {}).get("occupation") or []
    if profile.occupation and profile.occupation in targeted_occupations:
        weight *= 1.4
        reasons.insert(0, f"targeted at {profile.occupation}s")

    if benefit_type == "cash_transfer":
        weight *= 1.25
        reasons.append("guaranteed cash transfer, not contingent")

    return weight, reasons


def scheme_matching_advisor(
    profile: UserProfile,
    schemes: list[dict[str, Any]] | None = None,
    top_n: int = 5,
    include_possible: bool = True,
) -> dict[str, Any]:
    """
    Rank schemes this user should actually pursue. Pure and deterministic.
    """
    assessment = eligibility_advisor(profile, schemes)

    candidates = list(assessment["eligible"])
    if include_possible:
        candidates += assessment["possibly_eligible"]

    scored: list[dict[str, Any]] = []
    for row in candidates:
        benefit = row.get("benefit", {}) or {}
        annual_value = _annual_value(benefit)

        # log scale keeps a huge credit limit from dominating a small transfer
        benefit_weight = math.log10(annual_value + 10) / 6 if annual_value > 0 else 0.05

        need, reasons = _need_weight(row, profile)

        documents = len(row.get("documents", []) or [])
        effort_weight = max(MIN_EFFORT_WEIGHT, 1.0 - documents * EFFORT_PER_DOCUMENT)

        score = benefit_weight * need * effort_weight
        # An unconfirmed eligibility is worth less than a confirmed one.
        if row["verdict"] == "unknown":
            score *= 0.6

        scored.append({
            **row,
            "annual_value": round(annual_value, 2),
            "benefit_weight": round(benefit_weight, 4),
            "need_weight": round(need, 4),
            "effort_weight": round(effort_weight, 4),
            "match_score": round(min(score, 1.0) * 100, 2),
            "why": reasons or [f"{row.get('category', 'general')} support"],
        })

    scored.sort(key=lambda r: r["match_score"], reverse=True)
    top = scored[:top_n]

    total_annual = sum(r["annual_value"] for r in scored if r["verdict"] == "eligible")

    return {
        "schemes_evaluated": assessment["schemes_evaluated"],
        "eligible_count": assessment["eligible_count"],
        "possibly_eligible_count": assessment["possibly_eligible_count"],
        "matches": top,
        "all_scored": scored,
        "estimated_annual_benefit": round(total_annual, 2),
        "ask_user_for": assessment["ask_user_for"],
        "recommendations": [
            f"{r['name']} - {r['match_score']:.0f}% match, "
            f"about Rs {r['annual_value']:,.0f}/year"
            + (" (eligibility unconfirmed)" if r["verdict"] == "unknown" else "")
            for r in top
        ],
    }


def scheme_matching_node(
    state: FinancialState, schemes: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """LangGraph adapter."""
    return {"scheme_matching_result": scheme_matching_advisor(state["profile"], schemes)}
