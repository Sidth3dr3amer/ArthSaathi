"""
Behavioral Council -> Nudge Strategy Agent.

Turns detected patterns into specific interventions, and decides when to send
them.

Two constraints shape this agent:

1. **Nudges decay.** Sending five is worse than sending one, because attention
   is the scarce resource. The agent ranks and caps.
2. **A nudge is not a recommendation.** It changes the choice architecture --
   the default, the timing, the friction, the framing -- rather than telling
   someone what to do. The `mechanism` field records which lever is being
   pulled, so the strategy is auditable rather than a pile of notifications.

Timing is derived from the user's own observed cycle (salary day, month end),
because a nudge delivered at the moment of decision is worth several delivered
at a random Tuesday.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState

#: Choice-architecture levers, roughly ordered by observed effectiveness.
MECHANISMS = {
    "default": "Change what happens if the user does nothing",
    "friction": "Add a step before the unwanted action",
    "timing": "Deliver at the moment of decision",
    "framing": "Restate the same fact in more concrete terms",
    "salience": "Make an invisible cost visible",
    "commitment": "Get agreement in advance, when the decision is abstract",
    "social_proof": "Show what comparable people do",
}

#: How many nudges may be active at once. Attention is the binding constraint.
MAX_ACTIVE_NUDGES = 3


def _nudge(
    trigger: str, mechanism: str, message: str, timing: str,
    expected_value: float, channel: str = "app",
) -> dict[str, Any]:
    return {
        "trigger": trigger,
        "mechanism": mechanism,
        "mechanism_description": MECHANISMS.get(mechanism, ""),
        "message": message,
        "timing": timing,
        "channel": channel,
        "expected_annual_value": round(expected_value, 2),
    }


def nudge_strategy_advisor(
    bias_findings: list[dict[str, Any]] | None = None,
    habits: dict[str, Any] | None = None,
    emergency_status: str | None = None,
    monthly_surplus: float = 0.0,
) -> dict[str, Any]:
    """
    Build a ranked, capped nudge programme. Pure and deterministic.
    """
    findings = {f["bias"]: f for f in (bias_findings or [])}
    nudges: list[dict[str, Any]] = []

    if unsaved := findings.get("hyperbolic_discounting"):
        nudges.append(_nudge(
            trigger="salary_credited",
            mechanism="default",
            message=(
                f"Your salary just landed. Rs {max(monthly_surplus * 0.2, 0):,.0f} "
                "is scheduled to move to savings tonight unless you skip it."
            ),
            timing="within 2 hours of salary credit",
            expected_value=unsaved["estimated_annual_cost"] * 0.5,
        ))

    if creep := findings.get("status_quo_bias"):
        nudges.append(_nudge(
            trigger="quarterly_review",
            mechanism="salience",
            message=(
                f"Your subscriptions cost Rs {creep['estimated_annual_cost']:,.0f} "
                "a year. Here they are, ranked by how little you have used them."
            ),
            timing="first weekend of each quarter",
            expected_value=creep["estimated_annual_cost"] * 0.4,
        ))

    spikes = [
        f for f in (bias_findings or [])
        if f["bias"] in ("mental_accounting", "present_bias")
    ]
    if spikes:
        worst = max(spikes, key=lambda f: f["estimated_annual_cost"])
        category = worst["evidence"].get("category", "discretionary")
        window = worst["evidence"].get("window", "month_end")
        nudges.append(_nudge(
            trigger=f"{window}_window_opens",
            mechanism="timing",
            message=(
                f"Heads up: your {category.replace('_', ' ')} spending usually runs "
                f"{worst['evidence'].get('ratio', 0):.1f}x higher over the next few "
                f"days. Worth deciding a number now."
            ),
            timing=(
                "the 26th of each month" if window == "month_end"
                else "the morning after salary lands"
            ),
            expected_value=sum(f["estimated_annual_cost"] for f in spikes) * 0.35,
        ))

    if impulse := findings.get("impulse_buying"):
        nudges.append(_nudge(
            trigger="second_purchase_same_category_24h",
            mechanism="friction",
            message=(
                "That is your second order today. Adding it to a list until "
                "tomorrow instead?"
            ),
            timing="at checkout, on a repeat purchase within 24 hours",
            expected_value=impulse["estimated_annual_cost"] * 0.4,
        ))

    if inflation := findings.get("lifestyle_inflation"):
        nudges.append(_nudge(
            trigger="income_increase_detected",
            mechanism="commitment",
            message=(
                "Your income went up. Commit now to routing half the increase to "
                "savings -- easier to agree before the money arrives than after."
            ),
            timing="within a week of a detected raise",
            expected_value=inflation["estimated_annual_cost"] * 0.5,
        ))

    if emergency_status in ("Critical", "Vulnerable"):
        nudges.append(_nudge(
            trigger="emergency_fund_below_target",
            mechanism="framing",
            message=(
                "Your savings currently cover a few weeks of expenses. Each "
                "Rs 5,000 added buys roughly another three days of cover."
            ),
            timing="weekly, until three months of runway is reached",
            expected_value=0.0,
        ))

    if keystone := (habits or {}).get("keystone_habit"):
        # Only add the keystone reminder if nothing already fires on the same
        # cue. Two prompts on salary day saying the same thing is not twice the
        # nudge -- it is one nudge and one annoyance.
        anchor = (keystone.get("anchor") or "").lower()
        already_covered = any(
            anchor and anchor in n["timing"].lower() for n in nudges
        )
        if not already_covered:
            nudges.append(_nudge(
                trigger="keystone_habit_reminder",
                mechanism="commitment",
                message=keystone["implementation_intention"],
                timing=f"anchored to: {keystone.get('anchor') or 'the habit cue'}",
                expected_value=keystone["estimated_annual_value"] * 0.3,
            ))

    nudges.sort(key=lambda n: n["expected_annual_value"], reverse=True)

    # Cap by distinct trigger, so the active set never spends two of its three
    # slots on the same moment in the user's month.
    active: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []
    seen_triggers: set[str] = set()
    for nudge in nudges:
        if len(active) < MAX_ACTIVE_NUDGES and nudge["trigger"] not in seen_triggers:
            active.append(nudge)
            seen_triggers.add(nudge["trigger"])
        else:
            queued.append(nudge)

    return {
        "status": "Nudge programme built" if nudges else "No nudges indicated",
        "active_nudges": active,
        "queued_nudges": queued,
        "active_count": len(active),
        "suppressed_count": len(queued),
        "mechanisms_used": sorted({n["mechanism"] for n in active}),
        "expected_annual_value": round(
            sum(n["expected_annual_value"] for n in active), 2
        ),
        "cap_rationale": (
            f"Capped at {MAX_ACTIVE_NUDGES} active nudges: attention is the "
            "binding constraint, and additional prompts reduce the response rate "
            "of the ones that matter."
        ),
        "recommendations": [f"[{n['mechanism']}] {n['message']}" for n in active],
    }


def nudge_strategy_node(state: FinancialState) -> dict[str, Any]:
    """LangGraph adapter. Consumes Bias Detection and Habit Formation upstream."""
    bias = state.get("bias_detection_result") or {}
    habits = state.get("habit_formation_result") or {}
    emergency = state.get("emergency_fund_result") or {}
    profile = state["profile"]

    return {
        "nudge_strategy_result": nudge_strategy_advisor(
            bias_findings=bias.get("findings"),
            habits=habits,
            emergency_status=emergency.get("status"),
            monthly_surplus=profile.monthly_surplus,
        )
    }
