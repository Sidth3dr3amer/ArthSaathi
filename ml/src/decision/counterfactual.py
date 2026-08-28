"""
Decision Layer -> Counterfactual Simulator.

Answers "what if I did X instead?" by re-running the deterministic agent cores
against a modified copy of the profile and diffing the outcomes.

This is only possible because every agent core is a pure function of the
profile. Nothing here re-implements agent logic -- it perturbs an input, calls
the same code the live path calls, and compares. That means a counterfactual can
never drift from the real recommendation.

Scenarios are declarative so the UI and the LLM can both propose them without
touching this module.
"""

from __future__ import annotations

from typing import Any, Callable

from ..councils.cashflow.goal_allocation import goal_allocation_node
from ..councils.growth.retirement import retirement_node
from ..councils.risk.emergency_fund import emergency_fund_node
from ..schemas.profile import UserProfile
from ..schemas.state import FinancialState, new_state

#: Metrics worth diffing, as (result key, field, human label, higher_is_better).
TRACKED_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("emergency_fund_result", "completion_percent", "Emergency fund funded", True),
    ("emergency_fund_result", "months_to_goal", "Months to full runway", False),
    ("emergency_fund_result", "remaining_gap", "Emergency fund gap", False),
    ("goal_allocation_result", "shortfall", "Monthly goal shortfall", False),
    ("goal_allocation_result", "allocatable", "Allocatable surplus", True),
    ("retirement_result", "readiness_percent", "Retirement readiness", True),
    ("retirement_result", "additional_monthly_required", "Extra SIP needed", False),
)

#: The agents re-run for a counterfactual. Deliberately the cheap deterministic
#: ones -- no network, no LLM, so a scenario sweep stays instant.
SCENARIO_NODES: tuple[Callable[[FinancialState], dict[str, Any]], ...] = (
    emergency_fund_node,
    goal_allocation_node,
    retirement_node,
)


def _apply(profile: UserProfile, changes: dict[str, Any]) -> UserProfile:
    """Return a modified copy. Supports `+`/`-` prefixed relative deltas."""
    update: dict[str, Any] = {}
    for field, value in changes.items():
        if isinstance(value, str) and value and value[0] in "+-":
            current = getattr(profile, field, 0) or 0
            update[field] = current + float(value)
        else:
            update[field] = value
    return profile.model_copy(update=update)


def _evaluate(profile: UserProfile, query: str = "") -> FinancialState:
    state = new_state(profile, query)
    for node in SCENARIO_NODES:
        state.update(node(state))
    return state


def _diff(baseline: FinancialState, variant: FinancialState) -> list[dict[str, Any]]:
    rows = []
    for result_key, field, label, higher_is_better in TRACKED_METRICS:
        before = (baseline.get(result_key) or {}).get(field)
        after = (variant.get(result_key) or {}).get(field)
        if before is None or after is None:
            continue
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue

        delta = after - before
        if abs(delta) < 1e-9:
            direction = "unchanged"
        elif (delta > 0) == higher_is_better:
            direction = "better"
        else:
            direction = "worse"

        rows.append({
            "metric": label,
            "field": field,
            "before": round(before, 2),
            "after": round(after, 2),
            "delta": round(delta, 2),
            "percent_change": (
                round(delta / abs(before) * 100, 2) if before else None
            ),
            "direction": direction,
        })
    return rows


def counterfactual_advisor(
    profile: UserProfile,
    scenarios: dict[str, dict[str, Any]],
    query: str = "",
) -> dict[str, Any]:
    """
    Run each scenario against the same agents and diff against the baseline.

    `scenarios` maps a label to profile field changes, e.g.
        {"Cut dining by 5k": {"essential_expenses": "-5000"}}
    """
    baseline = _evaluate(profile, query)
    results: list[dict[str, Any]] = []

    for label, changes in scenarios.items():
        try:
            variant = _evaluate(_apply(profile, changes), query)
        except Exception as exc:
            results.append({
                "scenario": label, "changes": changes,
                "error": repr(exc), "metrics": [], "net_effect": "error",
            })
            continue

        metrics = _diff(baseline, variant)
        better = sum(1 for m in metrics if m["direction"] == "better")
        worse = sum(1 for m in metrics if m["direction"] == "worse")

        if better and not worse:
            net = "strictly better"
        elif worse and not better:
            net = "strictly worse"
        elif better or worse:
            net = "mixed"
        else:
            net = "no material change"

        results.append({
            "scenario": label,
            "changes": changes,
            "metrics": metrics,
            "improved": better,
            "worsened": worse,
            "net_effect": net,
        })

    ranked = sorted(
        results,
        key=lambda r: (r.get("improved", 0) - r.get("worsened", 0)),
        reverse=True,
    )

    return {
        "baseline": {
            label: (baseline.get(key) or {}).get(field)
            for key, field, label, _ in TRACKED_METRICS
            if (baseline.get(key) or {}).get(field) is not None
        },
        "scenarios_run": len(results),
        "results": results,
        "best_scenario": ranked[0]["scenario"] if ranked and ranked[0].get("improved") else None,
        "recommendations": [
            f"{r['scenario']}: {r['net_effect']} "
            f"({r.get('improved', 0)} metrics improved, {r.get('worsened', 0)} worsened)"
            for r in ranked
        ],
    }


def default_scenarios(profile: UserProfile) -> dict[str, dict[str, Any]]:
    """
    Scenarios worth asking about for most users, scaled to their own numbers.
    """
    income = profile.monthly_income or 0
    trim = round(max(income * 0.05, 1_000), -2)
    return {
        f"Cut monthly spending by Rs {trim:,.0f}": {"essential_expenses": f"-{trim}"},
        f"Increase income by Rs {trim * 2:,.0f}": {"monthly_income": f"+{trim * 2}"},
        "Clear all debt": {"debts": []},
        f"Add Rs {round(income, -3):,.0f} to the emergency fund": {
            "existing_emergency_fund": f"+{round(income, -3)}"
        },
    }


def counterfactual_node(
    state: FinancialState,
    scenarios: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """LangGraph adapter."""
    profile = state["profile"]
    scenarios = scenarios if scenarios is not None else default_scenarios(profile)
    return {
        "counterfactual_result": counterfactual_advisor(
            profile, scenarios, state.get("query", "")
        )
    }
