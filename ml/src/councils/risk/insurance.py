"""
Risk Council -> Insurance Agent.

New agent (not a migration). Sizes a user's protection gap across the three
covers that matter most in the Indian retail market, and prices the shortfall.

Method
------
Term life   Human Life Value, simplified: 10-15x annual income scaled by age
            (younger earners need a larger multiple -- more earning years to
            replace), plus outstanding debt, minus existing cover. Only material
            when someone depends on the income.
Health      A floater sized by dependants and age band. Medical inflation in
            India runs ahead of general inflation, so the bands are deliberately
            not generous.
Critical    Roughly one year of income once past 35, when incidence climbs.
illness

Premiums are indicative annual estimates per 1 lakh of cover, used to show the
cost of closing the gap -- not quotes.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

LAKH = 100_000

#: Term-cover multiple of annual income by age band. Younger => higher multiple.
TERM_MULTIPLE_BY_AGE = (
    (30, 15),
    (40, 12),
    (50, 10),
    (60, 7),
    (200, 5),
)

#: Base health floater by number of dependants, in lakh.
HEALTH_BASE_LAKH = {0: 5, 1: 7, 2: 10, 3: 12}
HEALTH_MAX_LAKH = 15

#: Indicative annual premium per lakh of cover (INR).
PREMIUM_PER_LAKH = {
    "term_life": 55,
    "health": 900,
    "critical_illness": 350,
}

#: Age loading applied to the indicative premium.
def _age_loading(age: int) -> float:
    if age < 30:
        return 1.0
    if age < 40:
        return 1.3
    if age < 50:
        return 1.9
    if age < 60:
        return 2.8
    return 4.0


def _term_multiple(age: int) -> int:
    for ceiling, multiple in TERM_MULTIPLE_BY_AGE:
        if age < ceiling:
            return multiple
    return 5


def insurance_advisor(
    monthly_income: float,
    age: int,
    dependents: int = 0,
    has_health_insurance: bool = True,
    has_life_insurance: bool = False,
    has_term_cover: bool = False,
    total_debt: float = 0.0,
    existing_term_cover: float = 0.0,
    existing_health_cover: float = 0.0,
) -> dict[str, Any]:
    """
    Size the protection gap. Pure and deterministic: no I/O, no LLM.

    Returns per-cover recommendations plus a single prioritised action.
    """
    annual_income = monthly_income * 12

    # ---- Term life: only meaningful when someone depends on the income ----
    if dependents > 0 or total_debt > 0:
        multiple = _term_multiple(age)
        term_required = annual_income * multiple + total_debt
    else:
        multiple = 0
        term_required = 0.0
    term_gap = max(term_required - existing_term_cover, 0.0)

    # ---- Health floater ----
    health_required = min(
        HEALTH_BASE_LAKH.get(min(dependents, 3), HEALTH_MAX_LAKH), HEALTH_MAX_LAKH
    ) * LAKH
    if age >= 45:
        health_required *= 1.5
    health_gap = max(health_required - existing_health_cover, 0.0)

    # ---- Critical illness ----
    critical_required = annual_income if age >= 35 else 0.0
    critical_gap = critical_required

    loading = _age_loading(age)
    covers = {
        "term_life": {
            "required": round(term_required, 2),
            "existing": round(existing_term_cover, 2),
            "gap": round(term_gap, 2),
            "income_multiple": multiple,
            "held": has_term_cover or has_life_insurance,
            "annual_premium_estimate": round(
                term_gap / LAKH * PREMIUM_PER_LAKH["term_life"] * loading, 2
            ),
        },
        "health": {
            "required": round(health_required, 2),
            "existing": round(existing_health_cover, 2),
            "gap": round(health_gap, 2),
            "held": has_health_insurance,
            "annual_premium_estimate": round(
                health_gap / LAKH * PREMIUM_PER_LAKH["health"] * loading, 2
            ),
        },
        "critical_illness": {
            "required": round(critical_required, 2),
            "existing": 0.0,
            "gap": round(critical_gap, 2),
            "held": False,
            "annual_premium_estimate": round(
                critical_gap / LAKH * PREMIUM_PER_LAKH["critical_illness"] * loading, 2
            ),
        },
    }

    # ---- Prioritise ----------------------------------------------------
    # Ranking on rupee gaps alone is wrong: term cover is denominated in crores
    # and health in lakhs, so term would always win on magnitude regardless of
    # urgency. Rank instead on *shortfall ratio* (how much of the needed cover is
    # missing, 0..1) times a criticality weight. Uninsured medical risk is the
    # most common cause of household financial ruin in India, so health carries
    # the highest weight -- but a mostly-covered health gap still yields to a
    # wide-open term gap, which is the behaviour we want.
    CRITICALITY = {"health": 3.0, "term_life": 2.0, "critical_illness": 1.0}

    gaps: list[tuple[str, float]] = []
    for name, required, gap in (
        ("health", health_required, health_gap),
        ("term_life", term_required, term_gap),
        ("critical_illness", critical_required, critical_gap),
    ):
        if gap <= 0:
            continue
        shortfall_ratio = gap / required if required > 0 else 0.0
        gaps.append((name, shortfall_ratio * CRITICALITY[name]))

    gaps.sort(key=lambda pair: pair[1], reverse=True)
    priority = gaps[0][0] if gaps else None

    total_gap = term_gap + health_gap + critical_gap
    total_premium = sum(c["annual_premium_estimate"] for c in covers.values())

    # 0..1 exposure: how much of the needed protection is missing, with an
    # explicit penalty for having no health cover at all.
    total_required = term_required + health_required + critical_required
    exposure = (total_gap / total_required) if total_required > 0 else 0.0
    if not has_health_insurance:
        exposure = min(1.0, exposure + 0.15)

    if exposure >= 0.75:
        status = "Severely Underinsured"
    elif exposure >= 0.5:
        status = "Underinsured"
    elif exposure >= 0.25:
        status = "Partially Covered"
    elif exposure > 0:
        status = "Adequately Covered"
    else:
        status = "Fully Covered"

    return {
        "covers": covers,
        "total_gap": round(total_gap, 2),
        "total_annual_premium_estimate": round(total_premium, 2),
        "monthly_premium_estimate": round(total_premium / 12, 2),
        "premium_as_income_percent": round(
            (total_premium / annual_income * 100) if annual_income > 0 else 0.0, 2
        ),
        "exposure": round(exposure, 2),
        "status": status,
        "priority_cover": priority,
        "recommendations": [
            f"Close the {name.replace('_', ' ')} gap of Rs {covers[name]['gap']:,.0f}"
            for name, _ in gaps
        ],
    }


def insurance_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter."""
    profile = state["profile"]
    result = insurance_advisor(
        monthly_income=profile.monthly_income,
        age=profile.age,
        dependents=profile.dependents,
        has_health_insurance=profile.has_health_insurance,
        has_life_insurance=profile.has_life_insurance,
        has_term_cover=profile.has_term_cover,
        total_debt=profile.total_debt,
    )
    return {"insurance_result": result}
