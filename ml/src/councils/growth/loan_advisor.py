"""
Growth Council -> Loan Advisor Agent.

New agent. Answers three separate questions that borrowers usually conflate:

  1. How much will a lender actually give me?      (FOIR-based headroom)
  2. How much should I borrow?                     (affordability, not maximum)
  3. Should I prepay what I have, or invest?       (rate arbitrage)

(1) and (2) differ, and the gap is where households get into trouble: banks
underwrite to roughly 50-55% FOIR, which leaves nothing for goals or shocks.
This agent reports the lender's ceiling and a prudent figure side by side.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

#: Fixed Obligation to Income Ratio ceilings.
LENDER_FOIR = 0.55        # what a bank will typically underwrite to
PRUDENT_FOIR = 0.40       # what leaves room for goals and shocks

#: Indicative annual rates by loan type (percent).
TYPICAL_RATES = {
    "home_loan": 8.5,
    "education_loan": 10.5,
    "car_loan": 9.5,
    "personal_loan": 14.0,
    "gold_loan": 11.0,
    "credit_card": 42.0,
    "bnpl": 24.0,
}

#: Typical maximum tenure in months.
TYPICAL_TENURE = {
    "home_loan": 240,
    "education_loan": 120,
    "car_loan": 84,
    "personal_loan": 60,
    "gold_loan": 24,
}

#: Assumed long-run post-tax return on invested surplus, for the prepay decision.
ASSUMED_INVESTMENT_RETURN = 11.0


def emi(principal: float, annual_rate: float, months: int) -> float:
    """Standard reducing-balance EMI."""
    if months <= 0:
        return 0.0
    r = annual_rate / 1200
    if r == 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def max_principal(affordable_emi: float, annual_rate: float, months: int) -> float:
    """Invert the EMI formula to get the borrowable principal."""
    if affordable_emi <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 1200
    if r == 0:
        return affordable_emi * months
    factor = (1 + r) ** months
    return affordable_emi * (factor - 1) / (r * factor)


def loan_advisor_advisor(
    monthly_income: float,
    existing_obligations: float = 0.0,
    loan_type: str = "personal_loan",
    requested_amount: float | None = None,
    tenure_months: int | None = None,
    annual_rate: float | None = None,
    investable_surplus: float = 0.0,
    highest_debt_rate: float | None = None,
) -> dict[str, Any]:
    """
    Assess borrowing capacity and the prepay-versus-invest trade-off.

    Pure and deterministic: no I/O, no LLM.
    """
    rate = annual_rate if annual_rate is not None else TYPICAL_RATES.get(loan_type, 14.0)
    months = tenure_months or TYPICAL_TENURE.get(loan_type, 60)

    lender_capacity_emi = max(monthly_income * LENDER_FOIR - existing_obligations, 0.0)
    prudent_capacity_emi = max(monthly_income * PRUDENT_FOIR - existing_obligations, 0.0)

    lender_max = max_principal(lender_capacity_emi, rate, months)
    prudent_max = max_principal(prudent_capacity_emi, rate, months)

    current_foir = (existing_obligations / monthly_income) if monthly_income > 0 else 0.0

    assessment: dict[str, Any] = {}
    if requested_amount:
        requested_emi = emi(requested_amount, rate, months)
        resulting_foir = (
            (existing_obligations + requested_emi) / monthly_income
            if monthly_income > 0 else 1.0
        )
        total_repayment = requested_emi * months
        assessment = {
            "requested_amount": round(requested_amount, 2),
            "monthly_emi": round(requested_emi, 2),
            "total_repayment": round(total_repayment, 2),
            "total_interest": round(total_repayment - requested_amount, 2),
            "resulting_foir": round(resulting_foir, 4),
            "within_lender_limit": requested_amount <= lender_max,
            "within_prudent_limit": requested_amount <= prudent_max,
        }
        if requested_amount > lender_max:
            assessment["verdict"] = "likely_rejected"
        elif requested_amount > prudent_max:
            assessment["verdict"] = "approvable_but_stretched"
        else:
            assessment["verdict"] = "affordable"

    # ---- Prepay vs invest -----------------------------------------------
    prepay: dict[str, Any] = {}
    if highest_debt_rate is not None and investable_surplus > 0:
        spread = highest_debt_rate - ASSUMED_INVESTMENT_RETURN
        prepay = {
            "highest_debt_rate": highest_debt_rate,
            "assumed_investment_return": ASSUMED_INVESTMENT_RETURN,
            "spread": round(spread, 2),
            "recommendation": "prepay_debt" if spread > 0 else "invest_surplus",
            "rationale": (
                f"Repaying debt at {highest_debt_rate:.1f}% is a guaranteed return "
                f"{abs(spread):.1f} points above the assumed "
                f"{ASSUMED_INVESTMENT_RETURN:.1f}% market return."
                if spread > 0 else
                f"Debt at {highest_debt_rate:.1f}% costs less than the assumed "
                f"{ASSUMED_INVESTMENT_RETURN:.1f}% return, so investing the surplus "
                f"is expected to leave you {abs(spread):.1f} points better off -- "
                f"though the market return is not guaranteed and the saving is."
            ),
        }

    if current_foir >= LENDER_FOIR:
        status = "Over-leveraged"
    elif current_foir >= PRUDENT_FOIR:
        status = "Stretched"
    elif current_foir >= 0.2:
        status = "Comfortable"
    else:
        status = "Ample headroom"

    return {
        "loan_type": loan_type,
        "annual_rate": rate,
        "tenure_months": months,
        "current_foir": round(current_foir, 4),
        "status": status,
        "lender_max_emi": round(lender_capacity_emi, 2),
        "prudent_max_emi": round(prudent_capacity_emi, 2),
        "lender_max_principal": round(lender_max, 2),
        "prudent_max_principal": round(prudent_max, 2),
        "assessment": assessment,
        "prepay_vs_invest": prepay,
        "recommendations": [
            f"A lender would likely approve up to Rs {lender_max:,.0f} "
            f"({months // 12}-year {loan_type.replace('_', ' ')} at {rate}%)",
            f"Borrowing beyond Rs {prudent_max:,.0f} leaves little room for goals or shocks",
        ],
    }


def loan_advisor_node(
    state: FinancialState,
    loan_type: str = "personal_loan",
    requested_amount: float | None = None,
) -> dict[str, Any]:
    """LangGraph adapter."""
    profile = state["profile"]
    obligations = sum(max(d.minimum_due, d.emi) for d in profile.debts)
    rates = [d.interest_rate for d in profile.debts if d.interest_rate]

    result = loan_advisor_advisor(
        monthly_income=profile.monthly_income,
        existing_obligations=obligations,
        loan_type=loan_type,
        requested_amount=requested_amount,
        investable_surplus=max(profile.monthly_surplus, 0.0),
        highest_debt_rate=max(rates) if rates else None,
    )
    return {"loan_advisor_result": result}
