"""
Risk Council -> Emergency Fund Agent.

Migrated verbatim from `EmergencyFundAdvisor/EmergencyFundAdvisor_FINAL.ipynb`.
The arithmetic is unchanged; only the node adapter was updated to read from the
unified `UserProfile` instead of flat state keys.

Model basis (from the original notebook's comments):
  * Buffer Stock Theory      -> buffer_ratio / urgency
  * Precautionary Saving     -> risk_score / risk_factor
  * Liquidity tiering        -> cash_target vs liquid_fund_target
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

BASE_MONTHS_MAP = {"govt": 6, "salaried": 6, "freelancer": 6, "business": 6}
JOB_RISK_MAP = {"govt": 1, "salaried": 2, "freelancer": 4, "business": 5}

# Share of total expenses treated as non-discretionary.
ESSENTIAL_EXPENSE_RATIO = 0.70


def emergency_fund_advisor(
    income: float,
    expenses: float,
    existing_emergency_fund: float,
    job_type: str,
    dependents: int = 0,
    has_health_insurance: bool = True,
) -> dict[str, Any]:
    """
    Size a user's emergency fund and the monthly contribution needed to reach it.

    Pure and deterministic: no I/O, no LLM. Safe to unit-test directly.
    """
    current_fund = existing_emergency_fund

    # ---- Target months: baseline, widened by dependants and missing cover ----
    target_months = (
        BASE_MONTHS_MAP.get(job_type, 6)
        + (2 if dependents >= 1 else 0)
        + (1 if dependents >= 3 else 0)
        + (3 if not has_health_insurance else 0)
    )

    essential_expenses = expenses * ESSENTIAL_EXPENSE_RATIO
    target_fund = target_months * essential_expenses
    remaining_gap = max(target_fund - current_fund, 0)

    completion = (
        min(100, round((current_fund / target_fund) * 100, 2))
        if target_fund > 0
        else 100
    )

    if completion < 25:
        status = "Critical"
    elif completion < 50:
        status = "Vulnerable"
    elif completion < 75:
        status = "Moderate"
    elif completion < 100:
        status = "Good"
    else:
        status = "Fully Prepared"

    # ---- Buffer Stock Theory ----
    buffer_ratio = min(current_fund / target_fund, 1) if target_fund > 0 else 1
    urgency = 1 - buffer_ratio

    # ---- Precautionary Saving Theory ----
    risk_score = JOB_RISK_MAP.get(job_type, 2)
    risk_score += min(dependents, 3)
    if not has_health_insurance:
        risk_score += 2
    risk_factor = min(risk_score / 10, 1)

    # ---- Savings capacity ----
    monthly_surplus = max(income - expenses, 0)
    savings_rate = monthly_surplus / income if income > 0 else 0

    priority_score = 0.8 * urgency + 0.2 * risk_factor

    emergency_allocation_pct = priority_score * savings_rate * 2
    emergency_allocation_pct = max(0.05, min(emergency_allocation_pct, 0.70))

    monthly_emergency_contribution = monthly_surplus * emergency_allocation_pct
    monthly_investment_contribution = monthly_surplus - monthly_emergency_contribution

    # ---- Liquidity tiers ----
    cash_target = min(essential_expenses * 2, target_fund * 0.50)
    liquid_fund_target = target_fund - cash_target

    months_to_goal = (
        round(remaining_gap / monthly_emergency_contribution, 1)
        if monthly_emergency_contribution > 0
        else None
    )

    return {
        "target_months": target_months,
        "essential_expenses": round(essential_expenses, 2),
        "target_emergency_fund": round(target_fund, 2),
        "current_emergency_fund": round(current_fund, 2),
        "remaining_gap": round(remaining_gap, 2),
        "completion_percent": completion,
        "status": status,
        "risk_factor": round(risk_factor, 2),
        "urgency": round(urgency, 2),
        "monthly_surplus": round(monthly_surplus, 2),
        "savings_rate": round(savings_rate, 2),
        "priority_score": round(priority_score, 2),
        "emergency_allocation_percent": round(emergency_allocation_pct * 100, 2),
        "monthly_emergency_contribution": round(monthly_emergency_contribution, 2),
        "monthly_investment_contribution": round(monthly_investment_contribution, 2),
        "cash_target": round(cash_target, 2),
        "liquid_fund_target": round(liquid_fund_target, 2),
        "months_to_goal": months_to_goal,
    }


def emergency_fund_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter. Reads the unified profile, writes one result key."""
    profile = state["profile"]
    result = emergency_fund_advisor(
        income=profile.monthly_income,
        expenses=profile.essential_expenses,
        existing_emergency_fund=profile.existing_emergency_fund,
        job_type=profile.job_type,
        dependents=profile.dependents,
        has_health_insurance=profile.has_health_insurance,
    )
    return {"emergency_fund_result": result}
