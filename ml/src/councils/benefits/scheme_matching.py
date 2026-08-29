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

#: What KIND of thing a benefit is. Summing these together is the single
#: easiest way to produce a nonsense number, and the first version of this
#: agent did exactly that: a street vendor earning Rs 2.16 lakh was told she
#: qualified for Rs 7.26 lakh a year -- 335% of her income -- because a
#: Rs 1 crore *loan* and a Rs 25 lakh business-subsidy *ceiling* were being
#: counted as annual income alongside a Rs 6,000 cash transfer.
#:
#: They are four different things and only the first is money arriving:
#:   income           cash or wages the user receives
#:   protection       risk transferred away; contingent, never income
#:   credit_access    capital they could borrow; a liability, not a benefit
#:   savings_capacity room to save their OWN money at a better rate
BENEFIT_KIND = {
    "cash_transfer": "income",
    "wage_employment": "income",
    "subsidy": "income",
    "interest_subsidy": "income",
    "insurance": "protection",
    "loan": "credit_access",
    "credit_line": "credit_access",
    "banking": "credit_access",
    "pension": "savings_capacity",
    "savings": "savings_capacity",
}

#: What fraction of the headline amount a user actually realises in a year.
#:
#: A naive reading values PMSBY -- a Rs 20/year accident policy -- at its
#: Rs 2,00,000 sum assured, ranking it above PM-KISAN's guaranteed Rs 6,000
#: transfer. An insurance payout is contingent; a cash transfer is not.
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

#: A one-off ceiling (PMEGP's Rs 25 lakh project subsidy, PMAY's Rs 2.67 lakh)
#: is a maximum a few applicants reach, not an annuity. Amortising it over five
#: years still overstates it badly, so headline-capped benefits are additionally
#: discounted and capped against what the user could plausibly absorb.
CEILING_TYPES = {"subsidy", "interest_subsidy", "loan", "credit_line"}
CEILING_DISCOUNT = 0.25


def _annual_value(benefit: dict[str, Any], annual_surplus: float | None = None) -> float:
    """
    Indicative annual rupee value the user actually realises.

    Combines payout frequency with a realisation factor, then applies two
    corrections that stop the figure becoming absurd:

    * headline CEILINGS are discounted, because a Rs 25 lakh maximum is what a
      few applicants reach, not what a typical one receives;
    * SAVINGS CAPACITY is capped by what the user can actually put aside. A
      PPF limit of Rs 1.5 lakh is worth nothing to someone with a Rs 3,000
      monthly surplus, and counting it as a benefit is telling them they have
      money they do not have.
    """
    amount = float(benefit.get("amount", 0) or 0)
    if amount <= 0:
        return 0.0

    kind_of = benefit.get("type", "")
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

    if kind_of in CEILING_TYPES:
        annualised *= CEILING_DISCOUNT

    # You cannot save more than you have spare.
    if BENEFIT_KIND.get(kind_of) == "savings_capacity" and annual_surplus is not None:
        annualised = min(annualised, max(annual_surplus, 0.0))

    return annualised * REALISATION.get(kind_of, 0.5)


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

    # What the user could plausibly set aside in a year, used to cap
    # savings-capacity benefits at something they can actually reach.
    annual_surplus = max(profile.monthly_surplus, 0.0) * 12

    scored: list[dict[str, Any]] = []
    for row in candidates:
        benefit = row.get("benefit", {}) or {}
        annual_value = _annual_value(benefit, annual_surplus)
        kind = BENEFIT_KIND.get(benefit.get("type", ""), "income")

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
            "benefit_kind": kind,
            "why": reasons or [f"{row.get('category', 'general')} support"],
        })

    scored.sort(key=lambda r: r["match_score"], reverse=True)
    top = scored[:top_n]

    # Sum by KIND. Adding a loan ceiling to a cash transfer produces a headline
    # figure that can exceed the user's income several times over, which is both
    # wrong and the kind of number that destroys trust the moment it is noticed.
    eligible = [r for r in scored if r["verdict"] == "eligible"]
    by_kind: dict[str, float] = {}
    for r in eligible:
        by_kind[r["benefit_kind"]] = by_kind.get(r["benefit_kind"], 0.0) + r["annual_value"]

    income_benefit = by_kind.get("income", 0.0)
    annual_income = (
        profile.annual_household_income
        or (profile.monthly_income * 12 if profile.monthly_income else 0.0)
    )

    return {
        "schemes_evaluated": assessment["schemes_evaluated"],
        "eligible_count": assessment["eligible_count"],
        "possibly_eligible_count": assessment["possibly_eligible_count"],
        "matches": top,
        "all_scored": scored,
        # Only money actually arriving. Credit access and savings headroom are
        # reported separately rather than folded into an income-like number.
        "estimated_annual_benefit": round(income_benefit, 2),
        "benefit_breakdown": {k: round(v, 2) for k, v in sorted(by_kind.items())},
        "credit_access": round(by_kind.get("credit_access", 0.0), 2),
        "protection_value": round(by_kind.get("protection", 0.0), 2),
        "savings_capacity": round(by_kind.get("savings_capacity", 0.0), 2),
        "benefit_as_income_percent": (
            round(income_benefit / annual_income * 100, 1) if annual_income else None
        ),
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
