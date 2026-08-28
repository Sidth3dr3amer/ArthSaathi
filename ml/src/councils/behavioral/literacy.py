"""
Behavioral Council -> Financial Literacy Agent.

Works out which concepts this user would benefit from understanding, inferred
from what they are actually doing rather than from a quiz.

The deck's premise is that literacy gaps show up as behaviour: someone paying
42% revolving interest while holding idle cash has a specific, identifiable gap,
and telling them "learn about credit" is useless next to showing them the
arithmetic on their own balance. So every lesson here is triggered by an
observed signal and carries the user's own numbers.

Lessons are ordered by what the gap is costing, and capped -- a curriculum of
twelve topics gets read by nobody.
"""

from __future__ import annotations

from typing import Any

from ...schemas.profile import UserProfile
from ...schemas.state import FinancialState

MAX_LESSONS = 5

#: Concept -> plain-language explanation. Deliberately written for someone with
#: no finance background, per the deck's literacy goal.
CONCEPTS: dict[str, dict[str, str]] = {
    "compound_interest_on_debt": {
        "title": "How credit card interest compounds",
        "plain": (
            "Card interest is charged monthly on the balance including previously "
            "charged interest. At 42% a year, an unpaid Rs 1,00,000 becomes about "
            "Rs 1,51,000 after twelve months if nothing is repaid."
        ),
        "why_it_matters": "It is almost always the most expensive rupee you owe.",
    },
    "emergency_fund": {
        "title": "Why an emergency fund comes before investing",
        "plain": (
            "An emergency fund is money kept deliberately boring and reachable, "
            "sized in months of essential expenses. Without it, an unexpected bill "
            "becomes debt at 40% or a forced sale of an investment at a bad price."
        ),
        "why_it_matters": "It is what stops a small shock becoming a large one.",
    },
    "term_insurance": {
        "title": "Term cover versus investment-linked policies",
        "plain": (
            "Term insurance pays out only if you die during the term and has no "
            "maturity value, which is exactly why it is cheap. Policies that mix "
            "insurance with investment usually do both jobs worse than buying them "
            "separately."
        ),
        "why_it_matters": "Cover for dependants at the lowest cost per rupee insured.",
    },
    "health_insurance": {
        "title": "Why health cover matters even when you are well",
        "plain": (
            "A single hospitalisation can cost several years of savings, and cover "
            "cannot be bought once you need it. Premiums also rise with age and any "
            "condition diagnosed before you buy is typically excluded."
        ),
        "why_it_matters": "Medical bills are a leading cause of household debt in India.",
    },
    "inflation": {
        "title": "What inflation does to money you are holding",
        "plain": (
            "At 6% inflation, money loses about half its purchasing power in twelve "
            "years. Cash in a savings account earning 3% is quietly shrinking in "
            "what it can buy, even though the number on the statement grows."
        ),
        "why_it_matters": "Doing nothing with money is itself a decision with a cost.",
    },
    "sip_and_rupee_cost_averaging": {
        "title": "Investing a fixed amount every month",
        "plain": (
            "Investing the same amount on a schedule buys more units when prices "
            "are low and fewer when high, which removes the need to guess timing. "
            "It also makes investing automatic rather than a monthly decision."
        ),
        "why_it_matters": "Consistency beats timing for almost every retail investor.",
    },
    "credit_score": {
        "title": "How a credit score is built",
        "plain": (
            "Scores are driven mostly by paying on time and by how much of your "
            "available limit you use. Using more than about 30% of a card limit "
            "lowers the score even when every payment is made on time."
        ),
        "why_it_matters": "It sets the interest rate you are offered on future loans.",
    },
    "debt_prioritisation": {
        "title": "Which debt to clear first",
        "plain": (
            "Clearing the highest-rate debt first minimises total interest paid. "
            "Repaying a 42% card is a guaranteed 42% return, which no investment "
            "reliably matches."
        ),
        "why_it_matters": "The order you repay in changes the total cost substantially.",
    },
    "government_schemes": {
        "title": "Entitlements you may already qualify for",
        "plain": (
            "Central and state schemes provide income support, subsidised credit "
            "and insurance to people who meet published criteria. They go unclaimed "
            "mostly because people do not know they qualify."
        ),
        "why_it_matters": "Money you are already entitled to costs nothing but paperwork.",
    },
    "lifestyle_inflation": {
        "title": "Why raises often do not feel like raises",
        "plain": (
            "Spending tends to expand to match income, so a higher salary can leave "
            "the same amount saved. Deciding in advance where a raise goes is what "
            "keeps it."
        ),
        "why_it_matters": "It determines whether earning more makes you better off.",
    },
    "subscription_costs": {
        "title": "The real cost of recurring charges",
        "plain": (
            "Small recurring charges feel trivial individually and are rarely "
            "reviewed. Six subscriptions at Rs 400 a month is Rs 28,800 a year."
        ),
        "why_it_matters": "Recurring costs compound quietly in a way one-off ones do not.",
    },
}


def _lesson(concept: str, trigger: str, personalised: str, cost: float = 0.0) -> dict[str, Any]:
    body = CONCEPTS[concept]
    return {
        "concept": concept,
        "title": body["title"],
        "explanation": body["plain"],
        "why_it_matters": body["why_it_matters"],
        "trigger": trigger,
        "personalised": personalised,
        "estimated_annual_cost_of_gap": round(cost, 2),
    }


def literacy_advisor(
    profile: UserProfile,
    bias_findings: list[dict[str, Any]] | None = None,
    emergency_status: str | None = None,
    scheme_count: int = 0,
) -> dict[str, Any]:
    """
    Identify literacy gaps from behaviour. Pure and deterministic.
    """
    findings = {f["bias"]: f for f in (bias_findings or [])}
    lessons: list[dict[str, Any]] = []

    # --- Expensive debt ---------------------------------------------------
    expensive = [d for d in profile.debts if (d.interest_rate or 0) >= 24]
    if expensive:
        worst = max(expensive, key=lambda d: d.interest_rate or 0)
        annual_interest = worst.outstanding_amount * (worst.interest_rate or 0) / 100
        lessons.append(_lesson(
            "compound_interest_on_debt",
            trigger=f"carrying {worst.name} at {worst.interest_rate:.0f}%",
            personalised=(
                f"Your {worst.name} balance of Rs {worst.outstanding_amount:,.0f} at "
                f"{worst.interest_rate:.0f}% costs about "
                f"Rs {annual_interest:,.0f} a year in interest alone."
            ),
            cost=annual_interest,
        ))
        if len(profile.debts) > 1:
            lessons.append(_lesson(
                "debt_prioritisation",
                trigger="more than one debt outstanding",
                personalised=(
                    f"You hold {len(profile.debts)} debts between "
                    f"{min(d.interest_rate or 0 for d in profile.debts):.0f}% and "
                    f"{max(d.interest_rate or 0 for d in profile.debts):.0f}%. "
                    "Clearing the highest rate first minimises total interest."
                ),
            ))

    # --- Protection gaps ---------------------------------------------------
    if not profile.has_health_insurance:
        lessons.append(_lesson(
            "health_insurance",
            trigger="no health cover recorded",
            personalised=(
                f"You have no health cover recorded"
                + (f" and {profile.dependents} dependants." if profile.dependents else ".")
            ),
        ))
    if profile.dependents > 0 and not (profile.has_life_insurance or profile.has_term_cover):
        lessons.append(_lesson(
            "term_insurance",
            trigger=f"{profile.dependents} dependants and no life cover",
            personalised=(
                f"{profile.dependents} people depend on your income and no life "
                "cover is recorded."
            ),
        ))

    # --- Runway -------------------------------------------------------------
    if emergency_status in ("Critical", "Vulnerable"):
        lessons.append(_lesson(
            "emergency_fund",
            trigger=f"emergency fund status is {emergency_status}",
            personalised=(
                f"Your emergency fund is Rs {profile.existing_emergency_fund:,.0f} "
                f"against essential expenses of Rs {profile.essential_expenses:,.0f} "
                "a month."
            ),
        ))

    # --- Behavioural gaps ---------------------------------------------------
    if inflation := findings.get("lifestyle_inflation"):
        lessons.append(_lesson(
            "lifestyle_inflation",
            trigger="discretionary spend growing faster than income",
            personalised=inflation["observation"],
            cost=inflation["estimated_annual_cost"],
        ))
    if creep := findings.get("status_quo_bias"):
        lessons.append(_lesson(
            "subscription_costs",
            trigger="subscriptions accumulating",
            personalised=creep["observation"],
            cost=creep["estimated_annual_cost"],
        ))
    if unsaved := findings.get("hyperbolic_discounting"):
        lessons.append(_lesson(
            "sip_and_rupee_cost_averaging",
            trigger="surplus not being captured",
            personalised=unsaved["observation"],
            cost=unsaved["estimated_annual_cost"],
        ))

    # --- Idle money ---------------------------------------------------------
    if profile.current_balance > profile.essential_expenses * 6 and not expensive:
        lessons.append(_lesson(
            "inflation",
            trigger="large idle balance",
            personalised=(
                f"Rs {profile.current_balance:,.0f} sitting in a current account is "
                "losing purchasing power at roughly 6% a year."
            ),
            cost=profile.current_balance * 0.03,
        ))

    # --- Unclaimed entitlements ---------------------------------------------
    if scheme_count > 0:
        lessons.append(_lesson(
            "government_schemes",
            trigger=f"{scheme_count} schemes matched",
            personalised=(
                f"You appear eligible for {scheme_count} government schemes that "
                "are not currently claimed."
            ),
        ))

    # --- Credit utilisation ---------------------------------------------------
    cards = [d for d in profile.debts if d.debt_type == "credit_card"]
    if cards and profile.monthly_income > 0:
        lessons.append(_lesson(
            "credit_score",
            trigger="revolving card balance",
            personalised=(
                f"You carry Rs {sum(c.outstanding_amount for c in cards):,.0f} on "
                "cards. High utilisation lowers your score even when payments are "
                "on time."
            ),
        ))

    lessons.sort(key=lambda l: l["estimated_annual_cost_of_gap"], reverse=True)
    selected = lessons[:MAX_LESSONS]

    coverage = 1 - (len(lessons) / len(CONCEPTS))
    if coverage >= 0.8:
        level = "Strong"
    elif coverage >= 0.6:
        level = "Developing"
    elif coverage >= 0.4:
        level = "Several gaps"
    else:
        level = "Many gaps"

    return {
        "status": f"{len(lessons)} literacy gap(s) identified",
        "literacy_level": level,
        "coverage_estimate": round(max(coverage, 0.0), 3),
        "gaps_identified": len(lessons),
        "curriculum": selected,
        "deferred": [l["concept"] for l in lessons[MAX_LESSONS:]],
        "total_cost_of_gaps": round(
            sum(l["estimated_annual_cost_of_gap"] for l in lessons), 2
        ),
        "recommendations": [f"{l['title']} - {l['personalised']}" for l in selected],
    }


def literacy_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter. Consumes Bias Detection, Emergency Fund and Benefits upstream."""
    bias = state.get("bias_detection_result") or {}
    emergency = state.get("emergency_fund_result") or {}
    schemes = state.get("scheme_matching_result") or {}

    return {
        "literacy_result": literacy_advisor(
            profile=state["profile"],
            bias_findings=bias.get("findings"),
            emergency_status=emergency.get("status"),
            scheme_count=schemes.get("eligible_count", 0),
        )
    }
