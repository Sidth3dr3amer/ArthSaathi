"""
Behavioral Council -> Habit Formation Agent.

Identifies what the user already does consistently, then proposes changes that
attach to those existing routines.

The premise is that new habits stick when they are anchored to an existing cue
rather than to willpower. So the agent first finds real, observed regularities
in the transaction history -- a salary credit on the 1st, a grocery run every
week -- and then writes each suggestion as an implementation intention
("when X happens, do Y"), which is the form the behavioural literature finds
most reliable.

Every proposed habit is scored on how much it is worth and how hard it is, and
only one is nominated as the keystone. Handing someone six new habits is the
same as handing them none.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState
from . import features as F

#: A merchant seen in at least this share of months is a settled routine.
CONSISTENCY_THRESHOLD = 0.6

#: Effort ratings, 1 (trivial) to 5 (hard).
EFFORT = {
    "automate_saving": 1,
    "cancel_subscription": 1,
    "move_payday_transfer": 1,
    "cooling_off_rule": 2,
    "weekly_review": 3,
    "cash_envelope": 4,
}


def _habit(
    name: str, cue: str, action: str, rationale: str,
    annual_value: float, effort: int, anchor: str | None = None,
) -> dict[str, Any]:
    # Value per unit of effort -- what makes a keystone worth choosing.
    leverage = annual_value / effort if effort else annual_value
    return {
        "name": name,
        "cue": cue,
        "action": action,
        "implementation_intention": f"When {cue}, {action}.",
        "rationale": rationale,
        "estimated_annual_value": round(annual_value, 2),
        "effort": effort,
        "leverage": round(leverage, 2),
        "anchor": anchor,
    }


def observed_routines(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Regularities already present in the user's behaviour."""
    months = max(feats["months"], 1)
    routines = []
    for merchant in feats["recurring"]:
        consistency = merchant["months_active"] / months
        if consistency < CONSISTENCY_THRESHOLD:
            continue
        routines.append({
            "merchant": merchant["merchant"],
            "category": merchant["category"],
            "consistency": round(consistency, 3),
            "months_active": merchant["months_active"],
            "monthly_amount": merchant["monthly_amount"],
            "annual_cost": merchant["annual_cost"],
            "strength": "strong" if consistency >= 0.9 else "moderate",
        })
    return routines


def habit_formation_advisor(
    transactions: list[dict[str, Any]],
    bias_findings: list[dict[str, Any]] | None = None,
    monthly_surplus: float = 0.0,
) -> dict[str, Any]:
    """
    Propose habits anchored to observed routines. Pure and deterministic.

    `bias_findings` comes from the Bias Detection agent when it ran upstream,
    so proposals target patterns actually observed rather than generic advice.
    """
    if not transactions:
        return {
            "status": "no transaction history",
            "routines": [],
            "proposed_habits": [],
            "keystone_habit": None,
            "recommendations": [],
        }

    feats = F.extract_features(transactions)
    routines = observed_routines(feats)
    findings = {f["bias"]: f for f in (bias_findings or [])}
    proposals: list[dict[str, Any]] = []

    income_months = sorted(feats["monthly_income"])
    has_regular_income = len(income_months) >= 3

    # --- Automate the surplus at source ----------------------------------
    unsaved = findings.get("hyperbolic_discounting")
    if unsaved and has_regular_income:
        value = unsaved["estimated_annual_cost"] * 0.5   # capture half, realistically
        proposals.append(_habit(
            name="Pay yourself first",
            cue="your salary lands",
            action="an automatic transfer moves a fixed amount to savings the same day",
            rationale=(
                "Surplus left in the spending account gets spent. Moving it on "
                "payday makes saving the default rather than a monthly decision."
            ),
            annual_value=value,
            effort=EFFORT["automate_saving"],
            anchor="salary credit",
        ))

    # --- Cancel what renews on inattention --------------------------------
    creep = findings.get("status_quo_bias")
    if creep:
        subs = [r for r in routines if r["category"] == "subscriptions"]
        proposals.append(_habit(
            name="Subscription audit",
            cue="the first weekend of each quarter arrives",
            action="open the subscriptions list and cancel anything unused since the last review",
            rationale=(
                f"{len(subs)} recurring charges renew automatically. A standing "
                "review converts an opt-out into an opt-in."
            ),
            annual_value=creep["estimated_annual_cost"] * 0.4,
            effort=EFFORT["cancel_subscription"],
            anchor="quarterly calendar",
        ))

    # --- Blunt the month-end and payday spikes ----------------------------
    spikes = [
        f for f in (bias_findings or [])
        if f["bias"] in ("mental_accounting", "present_bias")
    ]
    if spikes:
        worst = max(spikes, key=lambda f: f["estimated_annual_cost"])
        category = worst["evidence"].get("category", "discretionary")
        proposals.append(_habit(
            name=f"Cooling-off rule on {category.replace('_', ' ')}",
            cue=f"you are about to spend on {category.replace('_', ' ')} outside a planned occasion",
            action="add it to a list and revisit it the next day instead of buying immediately",
            rationale=(
                f"{category.replace('_', ' ').title()} spend runs "
                f"{worst['evidence'].get('ratio', 0):.1f}x higher in a predictable "
                "window. A one-day delay removes the impulse without banning the spend."
            ),
            annual_value=sum(f["estimated_annual_cost"] for f in spikes) * 0.4,
            effort=EFFORT["cooling_off_rule"],
            anchor=category,
        ))

    # --- Slow the impulse bursts ------------------------------------------
    impulse = findings.get("impulse_buying")
    if impulse:
        proposals.append(_habit(
            name="Remove one-tap checkout",
            cue="you open a shopping app",
            action="card details are not stored, so each purchase needs a deliberate step",
            rationale=(
                "Purchases arriving in consecutive-day bursts are usually one "
                "decision repeated. Friction at checkout interrupts the run."
            ),
            annual_value=impulse["estimated_annual_cost"] * 0.5,
            effort=EFFORT["automate_saving"],
            anchor="shopping apps",
        ))

    # --- Stop raises leaking into lifestyle -------------------------------
    inflation = findings.get("lifestyle_inflation")
    if inflation:
        proposals.append(_habit(
            name="Split every raise",
            cue="your income increases",
            action="route at least half of the increase straight to savings before adjusting spending",
            rationale=(
                "Discretionary spend is growing faster than income, so raises are "
                "being absorbed. Splitting at source keeps some of each raise."
            ),
            annual_value=inflation["estimated_annual_cost"] * 0.5,
            effort=EFFORT["move_payday_transfer"],
            anchor="salary revision",
        ))

    proposals.sort(key=lambda p: p["leverage"], reverse=True)
    keystone = proposals[0] if proposals else None

    return {
        "status": (
            "Habits proposed" if proposals else "No habit changes indicated"
        ),
        "months_analysed": feats["months"],
        "routines": routines,
        "strong_routines": [r for r in routines if r["strength"] == "strong"],
        "proposed_habits": proposals,
        "keystone_habit": keystone,
        "total_potential_value": round(
            sum(p["estimated_annual_value"] for p in proposals), 2
        ),
        "recommendations": (
            [f"Start with: {keystone['implementation_intention']}"] if keystone else []
        ) + [p["implementation_intention"] for p in proposals[1:3]],
    }


def habit_formation_node(
    state: FinancialState, transactions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """LangGraph adapter. Consumes Bias Detection's findings when available."""
    if transactions is None:
        transactions = state.get("transactions")
    if transactions is None:
        from ...common.synthetic import load_transactions

        transactions = load_transactions()

    bias = state.get("bias_detection_result") or {}
    profile = state["profile"]

    return {
        "habit_formation_result": habit_formation_advisor(
            transactions,
            bias_findings=bias.get("findings"),
            monthly_surplus=profile.monthly_surplus,
        )
    }
