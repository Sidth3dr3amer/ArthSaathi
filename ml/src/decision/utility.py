"""
Decision Layer -> Utility Optimizer.

Every council returns its own recommendations, each locally sensible and
competing for the same rupee. This module allocates the surplus across them.

Naive ranking by rupee value fails here, for two reasons:

  * Returns are not linear. The first rupee into an empty emergency fund is
    worth far more than the ten-thousandth, so each claim gets a diminishing
    marginal utility curve rather than a flat rate.
  * Some claims are not optional. Minimum debt service and health cover are
    prerequisites, not investments, and must be funded before anything that
    merely has a higher headline return.

So allocation runs in two passes: prerequisites first, then greedy allocation of
what remains, in small increments, always to whichever claim currently has the
highest marginal utility. That produces a split rather than winner-takes-all,
which is what a real plan looks like.
"""

from __future__ import annotations

import math
from typing import Any

from ..schemas.state import FinancialState

#: Rupee increment for greedy allocation. Small enough to produce a realistic
#: split, large enough to stay fast.
STEP = 500.0

#: Claims that must be funded before any optimisation, in priority order.
PREREQUISITE_ORDER = ("debt_service", "health_cover")


def _diminishing(base_utility: float, funded: float, saturation: float) -> float:
    """
    Marginal utility of the next rupee.

    Exponential decay toward saturation: the first rupee into an unmet need is
    worth `base_utility`, and the value falls as the need is met.
    """
    if saturation <= 0:
        return 0.0
    return base_utility * math.exp(-3.0 * funded / saturation)


def build_claims(state: FinancialState) -> list[dict[str, Any]]:
    """
    Turn council outputs into competing claims on the surplus.

    Only agents that actually ran contribute, so this works on a partial state
    from a narrowly-routed workflow.
    """
    profile = state["profile"]
    claims: list[dict[str, Any]] = []

    emergency = state.get("emergency_fund_result") or {}
    if emergency.get("remaining_gap", 0) > 0:
        claims.append({
            "claim": "emergency_fund",
            "label": "Build emergency runway",
            "base_utility": 1.0,
            "saturation": emergency["remaining_gap"],
            "kind": "protective",
            "rationale": (
                f"Runway is {emergency.get('completion_percent', 0)}% funded "
                f"({emergency.get('status')})."
            ),
        })

    debt = state.get("debt_trap_result") or {}
    if debt.get("total_debt", 0) > 0:
        rates = [d.interest_rate or 0 for d in profile.debts]
        worst = max(rates) if rates else 0
        claims.append({
            "claim": "debt_repayment",
            "label": "Repay high-cost debt",
            # A guaranteed 42% return outranks anything else available.
            "base_utility": min(worst / 30, 1.4),
            "saturation": debt["total_debt"],
            "kind": "guaranteed_return",
            "rationale": f"Highest debt rate is {worst:.0f}%, a guaranteed return.",
        })

    insurance = state.get("insurance_result") or {}
    if insurance.get("total_gap", 0) > 0:
        premium = insurance.get("total_annual_premium_estimate", 0)
        claims.append({
            "claim": "insurance",
            "label": "Close the protection gap",
            "base_utility": 1.2 if not profile.has_health_insurance else 0.7,
            "saturation": max(premium, 1.0),
            "kind": "protective",
            "rationale": (
                f"Protection status: {insurance.get('status')}, priority "
                f"{insurance.get('priority_cover')}."
            ),
        })

    goals = state.get("goal_allocation_result") or {}
    if goals.get("total_required_monthly", 0) > 0:
        claims.append({
            "claim": "goals",
            "label": "Fund stated goals",
            "base_utility": 0.6,
            "saturation": goals["total_required_monthly"],
            "kind": "aspirational",
            "rationale": f"Goals require Rs {goals['total_required_monthly']:,.0f}/month.",
        })

    retirement = state.get("retirement_result") or {}
    if retirement.get("additional_monthly_required", 0) > 0:
        claims.append({
            "claim": "retirement",
            "label": "Close the retirement gap",
            "base_utility": 0.65,
            "saturation": retirement["additional_monthly_required"],
            "kind": "long_term",
            "rationale": (
                f"Retirement readiness {retirement.get('readiness_percent', 0)}%."
            ),
        })

    return claims


def utility_advisor(
    claims: list[dict[str, Any]],
    surplus: float,
    mandatory_debt_service: float = 0.0,
    health_premium: float = 0.0,
) -> dict[str, Any]:
    """
    Allocate surplus across claims. Pure and deterministic.
    """
    surplus = max(surplus, 0.0)
    allocations = {c["claim"]: 0.0 for c in claims}
    prerequisites: list[dict[str, Any]] = []
    remaining = surplus

    # ---- Pass 1: prerequisites -----------------------------------------
    for label, amount, claim in (
        ("Minimum debt service", mandatory_debt_service, "debt_repayment"),
        ("Health cover premium", health_premium, "insurance"),
    ):
        if amount <= 0:
            continue
        funded = min(amount, remaining)
        remaining -= funded
        if claim in allocations:
            allocations[claim] += funded
        prerequisites.append({
            "label": label,
            "required": round(amount, 2),
            "funded": round(funded, 2),
            "fully_met": funded >= amount - 1e-9,
        })

    # ---- Pass 2: greedy marginal-utility allocation ---------------------
    steps = 0
    max_steps = 10_000                       # guard against a pathological loop
    while remaining >= STEP and claims and steps < max_steps:
        best, best_utility = None, 0.0
        for claim in claims:
            utility = _diminishing(
                claim["base_utility"], allocations[claim["claim"]], claim["saturation"]
            )
            if utility > best_utility:
                best, best_utility = claim, utility
        if best is None or best_utility <= 1e-6:
            break
        allocations[best["claim"]] += STEP
        remaining -= STEP
        steps += 1

    unallocated = remaining
    total_allocated = sum(allocations.values())

    plan = []
    for claim in claims:
        amount = allocations[claim["claim"]]
        if amount <= 0:
            continue
        plan.append({
            "claim": claim["claim"],
            "label": claim["label"],
            "kind": claim["kind"],
            "monthly_allocation": round(amount, 2),
            "share_of_surplus": round(amount / surplus, 4) if surplus else 0.0,
            "target": round(claim["saturation"], 2),
            "months_to_satisfy": (
                math.ceil(claim["saturation"] / amount) if amount > 0 else None
            ),
            "rationale": claim["rationale"],
        })
    plan.sort(key=lambda p: p["monthly_allocation"], reverse=True)

    return {
        "surplus": round(surplus, 2),
        "prerequisites": prerequisites,
        "allocation_plan": plan,
        "total_allocated": round(total_allocated, 2),
        "unallocated": round(unallocated, 2),
        "claims_considered": len(claims),
        "status": (
            "No surplus to allocate" if surplus <= 0
            else "No competing claims" if not claims
            else "Surplus allocated"
        ),
        "method": (
            "Prerequisites funded first, then greedy allocation by diminishing "
            f"marginal utility in Rs {STEP:,.0f} increments."
        ),
        "recommendations": [
            f"Rs {p['monthly_allocation']:,.0f}/month to {p['label'].lower()} "
            f"({p['share_of_surplus']:.0%} of surplus) - {p['rationale']}"
            for p in plan
        ],
    }


def utility_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter. Consumes whatever councils ran upstream."""
    profile = state["profile"]
    insurance = state.get("insurance_result") or {}

    mandatory = sum(max(d.minimum_due, d.emi) for d in profile.debts)
    health_premium = 0.0
    if not profile.has_health_insurance:
        covers = insurance.get("covers", {}) or {}
        health_premium = (covers.get("health", {}) or {}).get(
            "annual_premium_estimate", 0
        ) / 12

    return {
        "utility_result": utility_advisor(
            claims=build_claims(state),
            surplus=max(profile.monthly_surplus, 0.0) + mandatory,
            mandatory_debt_service=mandatory,
            health_premium=health_premium,
        )
    }
