"""
Growth Council -> Asset Allocation Agent.

New agent. Produces a target allocation across equity, debt, gold and cash.

Two-factor model
----------------
Risk CAPACITY  what the user can objectively absorb -- horizon, income
               stability, dependants, emergency runway, debt load. Computed,
               not asked.
Risk TOLERANCE what they say they want. Stated preference.

The binding constraint is capacity, not tolerance: someone with no emergency
fund and a 42% credit card should not hold 80% equity however bold they feel.
The agent reports both and allocates on the lower of the two, saying so.

Glide path starts from the (110 - age) equity heuristic rather than (100 - age),
which is closer to current Indian advisory practice given longer horizons, then
adjusts for capacity.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

#: Job stability contribution to risk capacity (0..1).
JOB_STABILITY = {
    "govt": 1.0,
    "salaried": 0.8,
    "business": 0.5,
    "freelancer": 0.4,
    "student": 0.3,
    "unsalaried": 0.3,
}

#: Floors and ceilings on equity, whatever the model says.
MIN_EQUITY = 0.0
MAX_EQUITY = 0.85

#: Baseline non-equity split once the equity share is fixed.
GOLD_SHARE_OF_REMAINDER = 0.15
CASH_SHARE_OF_REMAINDER = 0.20


def asset_allocation_advisor(
    age: int,
    monthly_income: float,
    job_type: str = "salaried",
    dependents: int = 0,
    emergency_fund_months: float = 0.0,
    debt_to_income: float = 0.0,
    has_high_interest_debt: bool = False,
    risk_tolerance: str = "moderate",
    investment_horizon_years: int | None = None,
    current_allocation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Recommend a target allocation. Pure and deterministic: no I/O, no LLM.
    """
    horizon = (
        investment_horizon_years
        if investment_horizon_years is not None
        else max(60 - age, 1)
    )

    # ---- Risk capacity (objective) -------------------------------------
    capacity = 0.0
    capacity += min(horizon / 30, 1.0) * 0.35              # time in market
    capacity += JOB_STABILITY.get(job_type, 0.5) * 0.25    # income stability
    capacity += min(emergency_fund_months / 6, 1.0) * 0.25  # runway
    capacity += max(0.0, 1 - dependents / 4) * 0.15         # obligations
    capacity -= min(debt_to_income, 0.5) * 0.4              # leverage drag
    capacity = max(0.0, min(1.0, capacity))

    tolerance_map = {"conservative": 0.3, "moderate": 0.6, "aggressive": 0.9}
    tolerance = tolerance_map.get(risk_tolerance, 0.6)

    binding = "capacity" if capacity < tolerance else "tolerance"
    effective_risk = min(capacity, tolerance)

    # ---- Equity share ---------------------------------------------------
    base_equity = max(0.0, (110 - age) / 100)
    equity = base_equity * (0.5 + effective_risk / 2)

    warnings: list[str] = []
    if has_high_interest_debt:
        # Clearing a 40%+ card is a risk-free return no portfolio can match.
        equity *= 0.5
        warnings.append(
            "High-interest debt outstanding: clearing it beats any expected "
            "market return, so equity is halved until it is repaid."
        )
    if emergency_fund_months < 3:
        equity *= 0.6
        warnings.append(
            f"Only {emergency_fund_months:.1f} months of runway: equity reduced "
            "until at least 3 months are in place."
        )

    equity = max(MIN_EQUITY, min(MAX_EQUITY, equity))

    remainder = 1.0 - equity
    gold = remainder * GOLD_SHARE_OF_REMAINDER
    cash = remainder * CASH_SHARE_OF_REMAINDER
    debt = remainder - gold - cash

    target = {
        "equity": round(equity, 4),
        "debt": round(debt, 4),
        "gold": round(gold, 4),
        "cash": round(cash, 4),
    }
    # Absorb rounding drift into the largest sleeve so the split always sums to 1.
    drift = round(1.0 - sum(target.values()), 4)
    if drift:
        largest = max(target, key=lambda k: target[k])
        target[largest] = round(target[largest] + drift, 4)

    # ---- Drift from current holdings ------------------------------------
    rebalancing: list[dict[str, Any]] = []
    if current_allocation:
        total = sum(current_allocation.values()) or 1.0
        for asset, target_share in target.items():
            current_share = current_allocation.get(asset, 0.0) / total
            delta = target_share - current_share
            if abs(delta) >= 0.05:
                rebalancing.append({
                    "asset": asset,
                    "current": round(current_share, 4),
                    "target": target_share,
                    "delta": round(delta, 4),
                    "action": "increase" if delta > 0 else "reduce",
                })
        rebalancing.sort(key=lambda r: abs(r["delta"]), reverse=True)

    if effective_risk >= 0.75:
        profile_label = "Aggressive"
    elif effective_risk >= 0.5:
        profile_label = "Balanced"
    elif effective_risk >= 0.3:
        profile_label = "Conservative"
    else:
        profile_label = "Capital Preservation"

    return {
        "risk_capacity": round(capacity, 3),
        "risk_tolerance": round(tolerance, 3),
        "effective_risk": round(effective_risk, 3),
        "binding_constraint": binding,
        "profile": profile_label,
        "horizon_years": horizon,
        "target_allocation": target,
        "target_allocation_percent": {k: round(v * 100, 2) for k, v in target.items()},
        "rebalancing": rebalancing,
        "warnings": warnings,
        "recommendations": [
            f"Hold {v * 100:.0f}% in {k}" for k, v in target.items() if v > 0
        ],
    }


def asset_allocation_node(state: FinancialState) -> dict[str, Any]:
    """
    LangGraph adapter.

    Consumes the Emergency Fund and Debt Trap agents when they ran upstream, so
    allocation reflects actual runway and leverage rather than assumptions.
    """
    profile = state["profile"]
    emergency = state.get("emergency_fund_result") or {}
    debt = state.get("debt_trap_result") or {}

    essential = profile.essential_expenses or 1.0
    runway_months = (
        emergency.get("current_emergency_fund", profile.existing_emergency_fund)
        / (essential * 0.7)
    ) if essential else 0.0

    high_interest = any(d.interest_rate and d.interest_rate >= 24 for d in profile.debts)

    result = asset_allocation_advisor(
        age=profile.age,
        monthly_income=profile.monthly_income,
        job_type=profile.job_type,
        dependents=profile.dependents,
        emergency_fund_months=round(runway_months, 2),
        debt_to_income=debt.get("debt_to_income", profile.debt_to_income),
        has_high_interest_debt=high_interest,
        risk_tolerance="aggressive" if profile.prefer_travel_perks else "moderate",
        current_allocation=profile.current_allocation or None,
    )
    return {"asset_allocation_result": result}
