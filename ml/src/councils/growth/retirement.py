"""
Growth Council -> Retirement Planning Agent.

New agent. Sizes the corpus a user needs at retirement and the monthly
contribution that closes the gap.

Method
------
1. Inflate today's essential expenses to the retirement date.
2. Size the corpus by the inflation-adjusted (real) return during drawdown,
   which is the honest way to handle a 25-30 year retirement -- a flat "25x
   expenses" rule silently assumes a 4% real withdrawal and breaks when
   inflation is 6% and debt yields 7%.
3. Project what the existing corpus and current contributions will grow to.
4. Solve the future-value-of-annuity for the additional monthly SIP needed.

All rates are configurable; the defaults are Indian long-run assumptions and are
stated in the output so a reviewer can disagree with them explicitly.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

DEFAULT_RETIREMENT_AGE = 60
DEFAULT_LIFE_EXPECTANCY = 85
DEFAULT_INFLATION = 6.0            # long-run CPI assumption
DEFAULT_PRE_RETIREMENT_RETURN = 11.0
DEFAULT_POST_RETIREMENT_RETURN = 7.0

#: Share of pre-retirement expenses still needed after retiring. Commuting and
#: child costs fall away; healthcare rises.
EXPENSE_REPLACEMENT_RATIO = 0.75


def _future_value(present: float, annual_rate: float, years: float) -> float:
    return present * (1 + annual_rate / 100) ** years


def _fv_of_sip(monthly: float, annual_rate: float, years: float) -> float:
    """Future value of a monthly contribution stream."""
    months = int(round(years * 12))
    if months <= 0 or monthly <= 0:
        return 0.0
    r = annual_rate / 1200
    if r == 0:
        return monthly * months
    return monthly * (((1 + r) ** months - 1) / r) * (1 + r)


def _sip_for_target(target: float, annual_rate: float, years: float) -> float:
    """Monthly contribution required to reach a target."""
    months = int(round(years * 12))
    if months <= 0 or target <= 0:
        return 0.0
    r = annual_rate / 1200
    if r == 0:
        return target / months
    return target / ((((1 + r) ** months - 1) / r) * (1 + r))


def retirement_advisor(
    age: int,
    monthly_expenses: float,
    current_corpus: float = 0.0,
    current_monthly_contribution: float = 0.0,
    retirement_age: int = DEFAULT_RETIREMENT_AGE,
    life_expectancy: int = DEFAULT_LIFE_EXPECTANCY,
    inflation: float = DEFAULT_INFLATION,
    pre_return: float = DEFAULT_PRE_RETIREMENT_RETURN,
    post_return: float = DEFAULT_POST_RETIREMENT_RETURN,
) -> dict[str, Any]:
    """
    Size the retirement corpus and the gap. Pure and deterministic.
    """
    years_to_retire = max(retirement_age - age, 0)
    years_in_retirement = max(life_expectancy - retirement_age, 1)

    if years_to_retire == 0:
        # Already at or past retirement age: report drawdown sustainability
        # rather than an accumulation plan.
        annual_need = monthly_expenses * 12 * EXPENSE_REPLACEMENT_RATIO
        real_return = (1 + post_return / 100) / (1 + inflation / 100) - 1
        sustainable = current_corpus * real_return + current_corpus / years_in_retirement
        return {
            "phase": "drawdown",
            "years_to_retire": 0,
            "years_in_retirement": years_in_retirement,
            "annual_need_today": round(annual_need, 2),
            "current_corpus": round(current_corpus, 2),
            "sustainable_annual_withdrawal": round(max(sustainable, 0.0), 2),
            "shortfall": round(max(annual_need - sustainable, 0.0), 2),
            "on_track": sustainable >= annual_need,
            "assumptions": {
                "inflation": inflation,
                "post_retirement_return": post_return,
                "life_expectancy": life_expectancy,
            },
            "recommendations": [],
        }

    # ---- 1. Expenses at retirement --------------------------------------
    monthly_need_at_retirement = _future_value(
        monthly_expenses * EXPENSE_REPLACEMENT_RATIO, inflation, years_to_retire
    )
    annual_need_at_retirement = monthly_need_at_retirement * 12

    # ---- 2. Corpus required ---------------------------------------------
    # Real return during drawdown; a present-value annuity over the retirement span.
    real_return = (1 + post_return / 100) / (1 + inflation / 100) - 1
    if abs(real_return) < 1e-9:
        required_corpus = annual_need_at_retirement * years_in_retirement
    else:
        required_corpus = annual_need_at_retirement * (
            (1 - (1 + real_return) ** -years_in_retirement) / real_return
        )

    # ---- 3. Projected corpus --------------------------------------------
    projected_existing = _future_value(current_corpus, pre_return, years_to_retire)
    projected_contributions = _fv_of_sip(
        current_monthly_contribution, pre_return, years_to_retire
    )
    projected_corpus = projected_existing + projected_contributions

    gap = max(required_corpus - projected_corpus, 0.0)

    # ---- 4. Additional SIP required --------------------------------------
    additional_sip = _sip_for_target(gap, pre_return, years_to_retire)

    readiness = (
        min(projected_corpus / required_corpus, 1.0) if required_corpus > 0 else 1.0
    )
    if readiness >= 1.0:
        status = "On track"
    elif readiness >= 0.75:
        status = "Slightly behind"
    elif readiness >= 0.4:
        status = "Behind"
    else:
        status = "Critically behind"

    return {
        "phase": "accumulation",
        "years_to_retire": years_to_retire,
        "years_in_retirement": years_in_retirement,
        "monthly_need_at_retirement": round(monthly_need_at_retirement, 2),
        "required_corpus": round(required_corpus, 2),
        "current_corpus": round(current_corpus, 2),
        "projected_corpus": round(projected_corpus, 2),
        "projected_from_existing": round(projected_existing, 2),
        "projected_from_contributions": round(projected_contributions, 2),
        "gap": round(gap, 2),
        "current_monthly_contribution": round(current_monthly_contribution, 2),
        "additional_monthly_required": round(additional_sip, 2),
        "total_monthly_required": round(
            current_monthly_contribution + additional_sip, 2
        ),
        "readiness": round(readiness, 4),
        "readiness_percent": round(readiness * 100, 2),
        "status": status,
        "on_track": readiness >= 1.0,
        "assumptions": {
            "inflation": inflation,
            "pre_retirement_return": pre_return,
            "post_retirement_return": post_return,
            "retirement_age": retirement_age,
            "life_expectancy": life_expectancy,
            "expense_replacement_ratio": EXPENSE_REPLACEMENT_RATIO,
        },
        "recommendations": (
            [
                f"Invest an additional Rs {additional_sip:,.0f}/month to close a "
                f"Rs {gap:,.0f} corpus gap over {years_to_retire} years"
            ]
            if gap > 0
            else ["Current contributions are projected to meet the target corpus"]
        ),
    }


def retirement_node(state: FinancialState) -> dict[str, Any]:
    """
    LangGraph adapter.

    Uses the goal allocation agent's leftover surplus as the current retirement
    contribution when it ran upstream, so the two agents do not double-count the
    same rupee.
    """
    profile = state["profile"]
    allocation = state.get("goal_allocation_result") or {}

    # Prefer the surplus the goal agent left unallocated, so the two agents do
    # not both claim the same rupee. Fall back to the profile's stated figure.
    contribution = profile.monthly_investment
    if allocation:
        allocated_to_goals = sum(
            g.get("allocated_monthly", 0) for g in allocation.get("goals", [])
        )
        leftover = max(allocation.get("allocatable", 0.0) - allocated_to_goals, 0.0)
        contribution = max(contribution, leftover)

    result = retirement_advisor(
        age=profile.age,
        monthly_expenses=profile.essential_expenses,
        # `existing_emergency_fund` is runway, not a retirement asset.
        current_corpus=profile.retirement_corpus,
        current_monthly_contribution=contribution,
    )
    return {"retirement_result": result}
