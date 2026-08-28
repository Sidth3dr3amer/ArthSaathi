"""
Cashflow Council -> Goal Allocation Agent.

New agent (not a migration). Splits a user's monthly surplus across their goals
and reports honestly when the goals do not fit the surplus.

The interesting decision here is what to do when goals are unaffordable. Two
options are reported rather than one, because they are different trade-offs the
user must choose between:

  * `required_extension` -- keep every target amount, push the deadlines out
  * `feasible_targets`   -- keep every deadline, reduce the target amounts

Allocation itself is priority-weighted, and the emergency fund is treated as a
prior claim on surplus: goals get what is left after runway is secured, because
funding a holiday while one month from insolvency is not advice worth giving.
"""

from __future__ import annotations

import math
from typing import Any

from ...schemas.state import FinancialState

PRIORITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}

#: Share of surplus reserved for the emergency fund while it is below target.
EMERGENCY_FIRST_SHARE = 0.5


def goal_allocation_advisor(
    goals: list[dict[str, Any]],
    monthly_surplus: float,
    emergency_gap: float = 0.0,
    emergency_monthly_contribution: float = 0.0,
) -> dict[str, Any]:
    """
    Allocate surplus across goals. Pure and deterministic: no I/O, no LLM.

    `goals` entries use the `Goal` schema fields: name, target_amount,
    current_amount, target_months, priority.
    """
    surplus = max(monthly_surplus, 0.0)

    # Emergency runway has first claim on surplus while it is short.
    reserved = 0.0
    if emergency_gap > 0:
        reserved = min(
            surplus * EMERGENCY_FIRST_SHARE,
            emergency_monthly_contribution or surplus * EMERGENCY_FIRST_SHARE,
        )
    allocatable = max(surplus - reserved, 0.0)

    enriched: list[dict[str, Any]] = []
    total_required = 0.0
    for goal in goals:
        target = float(goal.get("target_amount", 0) or 0)
        current = float(goal.get("current_amount", 0) or 0)
        months = int(goal.get("target_months", 0) or 0)
        priority = str(goal.get("priority", "medium")).lower()

        remaining = max(target - current, 0.0)
        required = (remaining / months) if months > 0 else remaining
        total_required += required

        enriched.append({
            "name": goal.get("name", "unnamed goal"),
            "target_amount": round(target, 2),
            "current_amount": round(current, 2),
            "remaining": round(remaining, 2),
            "target_months": months,
            "priority": priority,
            "weight": PRIORITY_WEIGHT.get(priority, 2.0),
            "required_monthly": round(required, 2),
            "progress_percent": round(min(current / target * 100, 100), 2) if target > 0 else 100.0,
        })

    feasible = total_required <= allocatable
    total_weight = sum(g["weight"] * (1 if g["remaining"] > 0 else 0) for g in enriched)

    for goal in enriched:
        if goal["remaining"] <= 0:
            goal["allocated_monthly"] = 0.0
            goal["funded_percent_of_need"] = 100.0
            goal["months_to_goal"] = 0
            goal["on_track"] = True
            continue

        if feasible:
            allocated = goal["required_monthly"]
            # Funding the required amount hits the deadline by construction.
            # Deriving this from the rounded allocation instead would tip a
            # fully-funded goal one month past its target.
            months_to_goal = goal["target_months"] or math.ceil(
                goal["remaining"] / allocated
            )
        else:
            if total_weight > 0:
                allocated = allocatable * (goal["weight"] / total_weight)
            else:
                allocated = 0.0
            months_to_goal = (
                math.ceil(goal["remaining"] / allocated) if allocated > 0 else None
            )

        goal["allocated_monthly"] = round(allocated, 2)
        goal["funded_percent_of_need"] = round(
            (allocated / goal["required_monthly"] * 100) if goal["required_monthly"] > 0 else 100.0,
            2,
        )
        goal["months_to_goal"] = months_to_goal
        goal["on_track"] = (
            goal["months_to_goal"] is not None
            and goal["target_months"] > 0
            and goal["months_to_goal"] <= goal["target_months"]
        )

    shortfall = max(total_required - allocatable, 0.0)

    # Two ways out of an infeasible plan -- the user picks which to give up.
    required_extension: list[dict[str, Any]] = []
    feasible_targets: list[dict[str, Any]] = []
    if not feasible:
        for goal in enriched:
            if goal["remaining"] <= 0:
                continue
            allocated = goal["allocated_monthly"]
            if allocated > 0:
                required_extension.append({
                    "name": goal["name"],
                    "original_months": goal["target_months"],
                    "required_months": goal["months_to_goal"],
                })
            feasible_targets.append({
                "name": goal["name"],
                "original_target": goal["target_amount"],
                "feasible_target": round(
                    goal["current_amount"] + allocated * goal["target_months"], 2
                ),
            })

    if not goals:
        status = "No goals set"
    elif feasible:
        status = "All goals fundable"
    elif allocatable <= 0:
        status = "No surplus to allocate"
    else:
        status = "Goals exceed surplus"

    return {
        "monthly_surplus": round(surplus, 2),
        "reserved_for_emergency": round(reserved, 2),
        "allocatable": round(allocatable, 2),
        "total_required_monthly": round(total_required, 2),
        "shortfall": round(shortfall, 2),
        "feasible": feasible,
        "status": status,
        "goals": enriched,
        "required_extension": required_extension,
        "feasible_targets": feasible_targets,
        "recommendations": [
            (
                f"Allocate Rs {g['allocated_monthly']:,.0f}/month to {g['name']}"
                + ("" if g["on_track"] else
                   f" (needs {g['months_to_goal']} months vs {g['target_months']} planned)")
            )
            for g in enriched if g["remaining"] > 0
        ],
    }


def goal_allocation_node(state: FinancialState) -> dict[str, Any]:
    """
    LangGraph adapter.

    Consumes the Emergency Fund agent's output when it ran earlier in the graph,
    so runway is reserved before goals are funded.
    """
    profile = state["profile"]
    emergency = state.get("emergency_fund_result") or {}

    result = goal_allocation_advisor(
        goals=[g.model_dump() for g in profile.goals],
        monthly_surplus=profile.monthly_surplus,
        emergency_gap=float(emergency.get("remaining_gap", 0) or 0),
        emergency_monthly_contribution=float(
            emergency.get("monthly_emergency_contribution", 0) or 0
        ),
    )
    return {"goal_allocation_result": result}
