"""
Cashflow Council -> Expense Optimizer Agent.

New agent (not a migration). Benchmarks each spend category against a healthy
share of income, quantifies the overshoot, and ranks categories by how much is
realistically recoverable.

Two ideas keep this honest rather than preachy:

  * Categories differ in how compressible they are. Rent cannot be cut this
    month; dining can. `RECOVERABLE_SHARE` encodes that, so the headline savings
    figure is achievable rather than theoretical.
  * Benchmarks are shares of *income*, not of spend, so a user who is simply
    living within their means is not told to optimise anything.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

#: Healthy ceiling as a share of monthly income, and how much of any overshoot
#: is realistically recoverable within a month or two.
BENCHMARKS: dict[str, tuple[float, float]] = {
    # category            ceiling  recoverable
    "rent":               (0.30,   0.10),
    "housing":            (0.30,   0.10),
    "utility_bills":      (0.06,   0.25),
    "groceries":          (0.12,   0.30),
    "transport":          (0.08,   0.35),
    "fuel":               (0.06,   0.35),
    "dining":             (0.06,   0.70),
    "entertainment":      (0.04,   0.75),
    "online_shopping":    (0.06,   0.70),
    "offline_retail":     (0.05,   0.65),
    "subscriptions":      (0.02,   0.85),
    "travel":             (0.07,   0.60),
    "international":      (0.03,   0.70),
    "education":          (0.10,   0.10),
    "healthcare":         (0.05,   0.05),
    "insurance":          (0.06,   0.05),
    "others":             (0.05,   0.50),
}

#: Applied to any category not in the table.
DEFAULT_BENCHMARK = (0.05, 0.40)


def expense_optimizer_advisor(
    monthly_income: float,
    monthly_spend: dict[str, float],
    essential_expenses: float = 0.0,
) -> dict[str, Any]:
    """
    Find recoverable overspend. Pure and deterministic: no I/O, no LLM.
    """
    total_spend = sum(monthly_spend.values())

    if monthly_income <= 0 or not monthly_spend:
        return {
            "total_spend": round(total_spend, 2),
            "spend_ratio": 0.0,
            "categories": [],
            "overspending": [],
            "potential_monthly_savings": 0.0,
            "potential_annual_savings": 0.0,
            "status": "Insufficient data",
            "recommendations": [],
        }

    categories: list[dict[str, Any]] = []
    for name, amount in monthly_spend.items():
        ceiling_share, recoverable_share = BENCHMARKS.get(name, DEFAULT_BENCHMARK)
        benchmark = monthly_income * ceiling_share
        overshoot = max(amount - benchmark, 0.0)
        recoverable = overshoot * recoverable_share

        categories.append({
            "category": name,
            "amount": round(amount, 2),
            "income_share": round(amount / monthly_income, 4),
            "benchmark": round(benchmark, 2),
            "benchmark_share": ceiling_share,
            "overshoot": round(overshoot, 2),
            "recoverable": round(recoverable, 2),
            "over_benchmark": overshoot > 0,
        })

    categories.sort(key=lambda c: c["recoverable"], reverse=True)
    overspending = [c for c in categories if c["over_benchmark"]]

    potential = sum(c["recoverable"] for c in categories)
    spend_ratio = total_spend / monthly_income

    if spend_ratio >= 1.0:
        status = "Spending exceeds income"
    elif spend_ratio >= 0.9:
        status = "Critically tight"
    elif spend_ratio >= 0.75:
        status = "Tight"
    elif spend_ratio >= 0.5:
        status = "Comfortable"
    else:
        status = "Highly efficient"

    recommendations = [
        f"Trim {c['category'].replace('_', ' ')} by about "
        f"Rs {c['recoverable']:,.0f}/month "
        f"(currently {c['income_share'] * 100:.1f}% of income vs "
        f"{c['benchmark_share'] * 100:.0f}% benchmark)"
        for c in overspending[:5]
    ]

    return {
        "total_spend": round(total_spend, 2),
        "spend_ratio": round(spend_ratio, 4),
        "categories": categories,
        "overspending": overspending,
        "potential_monthly_savings": round(potential, 2),
        "potential_annual_savings": round(potential * 12, 2),
        "savings_as_income_percent": round(potential / monthly_income * 100, 2),
        "status": status,
        "recommendations": recommendations,
    }


def expense_optimizer_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter."""
    profile = state["profile"]
    spend = dict(profile.monthly_spend)

    # Fall back to a single aggregate bucket when no category breakdown exists,
    # so the agent still reports the income/spend ratio rather than nothing.
    if not spend and profile.essential_expenses:
        spend = {"others": profile.essential_expenses}

    result = expense_optimizer_advisor(
        monthly_income=profile.monthly_income,
        monthly_spend=spend,
        essential_expenses=profile.essential_expenses,
    )
    return {"expense_optimizer_result": result}
