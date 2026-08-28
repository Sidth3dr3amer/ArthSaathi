"""
The "Update Memory" step.

Every workflow diagram in the deck terminates in a Memory Update box. This module
is that box: it turns the agent results sitting in `FinancialState` into typed,
embedded memories, and provides the matching recall node for the front of a
workflow.

Design notes
------------
* Each agent's result is summarised into ONE human-readable sentence, because
  that sentence is what gets embedded and later matched against a user's
  question. Dumping raw JSON into the embedding would bury the signal.
* The routing table below decides which of the six memory types each agent
  writes to, so recall can be scoped ("what do we know about their habits?").
* Writes never raise into a workflow: a memory failure is recorded in
  `state["errors"]` rather than aborting a recommendation the user is waiting on.
"""

from __future__ import annotations

from typing import Any, Callable

from ..schemas.state import RESULT_KEYS, FinancialState
from .store import MemoryStore, get_store

#: Which memory type each agent's output belongs in.
AGENT_MEMORY_TYPE: dict[str, str] = {
    "emergency_fund_result": "semantic",
    "insurance_result": "semantic",
    "debt_trap_result": "semantic",
    "fraud_result": "episodic",
    "asset_allocation_result": "semantic",
    "credit_card_result": "semantic",
    "loan_advisor_result": "semantic",
    "retirement_result": "goal",
    "scheme_matching_result": "semantic",
    "eligibility_result": "semantic",
    "bias_detection_result": "behavioral",
    "habit_formation_result": "behavioral",
    "nudge_strategy_result": "behavioral",
    "literacy_result": "behavioral",
    "stability_result": "simulation",
    "income_projection_result": "simulation",
    "expense_optimizer_result": "semantic",
    "goal_allocation_result": "goal",
}


def _money(value: Any) -> str:
    try:
        return f"Rs {float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


#: One-sentence summarisers. Anything without an entry falls back to a generic line.
SUMMARISERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "emergency_fund_result": lambda r: (
        f"Emergency fund is {r.get('completion_percent')}% funded "
        f"({r.get('status')}); target {_money(r.get('target_emergency_fund'))}, "
        f"gap {_money(r.get('remaining_gap'))}."
    ),
    "debt_trap_result": lambda r: (
        f"Debt position: {r.get('scenario')}, total {_money(r.get('total_debt'))}, "
        f"debt-to-income {r.get('debt_to_income')}."
    ),
    "fraud_result": lambda r: (
        f"Fraud check on {r.get('company_name') or 'an unnamed offer'}: "
        f"risk {r.get('risk_score')} ({r.get('risk_level')})."
    ),
    "credit_card_result": lambda r: (
        "Card recommendation: "
        + (
            ", ".join(
                f"{rec['card'].get('card_name')} (net {_money(rec.get('net_value'))})"
                for rec in r.get("recommendations", [])[:2]
            )
            or "no eligible cards"
        )
        + "."
    ),
    "stability_result": lambda r: (
        f"Cashflow projection over {r.get('months_ahead')} months: risk "
        f"{(r.get('risk') or {}).get('score')} "
        f"({(r.get('risk') or {}).get('rating')}), minimum balance "
        f"{_money((r.get('risk') or {}).get('min_projected_balance'))}."
    ),
    "income_projection_result": lambda r: (
        f"Income projected by {r.get('method')}: "
        f"{', '.join(_money(v) for v in (r.get('forecast') or [])[:3])} ..."
    ),
    "insurance_result": lambda r: (
        f"Insurance status {r.get('status')}: total protection gap "
        f"{_money(r.get('total_gap'))}, priority cover "
        f"{(r.get('priority_cover') or 'none').replace('_', ' ')}, "
        f"estimated premium {_money(r.get('monthly_premium_estimate'))}/month."
    ),
    "asset_allocation_result": lambda r: (
        f"Asset allocation profile {r.get('profile')} (bound by "
        f"{r.get('binding_constraint')}): "
        + ", ".join(
            f"{k} {v}%" for k, v in (r.get("target_allocation_percent") or {}).items()
        )
        + "."
    ),
    "loan_advisor_result": lambda r: (
        f"Borrowing capacity {r.get('status')}: FOIR {r.get('current_foir')}, "
        f"a lender would allow up to {_money(r.get('lender_max_principal'))}, "
        f"prudent limit {_money(r.get('prudent_max_principal'))}."
        + (
            f" Advice on surplus: {(r.get('prepay_vs_invest') or {}).get('recommendation')}."
            if r.get("prepay_vs_invest") else ""
        )
    ),
    "retirement_result": lambda r: (
        f"Retirement drawdown: corpus {_money(r.get('current_corpus'))}, "
        f"sustainable withdrawal {_money(r.get('sustainable_annual_withdrawal'))}/year."
        if r.get("phase") == "drawdown" else
        f"Retirement {r.get('status')} at {r.get('readiness_percent')}% readiness; "
        f"needs a corpus of {_money(r.get('required_corpus'))} in "
        f"{r.get('years_to_retire')} years, requiring an extra "
        f"{_money(r.get('additional_monthly_required'))}/month."
    ),
    "expense_optimizer_result": lambda r: (
        f"Spending is {r.get('status')} at {round((r.get('spend_ratio') or 0) * 100)}% "
        f"of income; about {_money(r.get('potential_monthly_savings'))}/month is "
        "recoverable"
        + (
            " from " + ", ".join(
                c["category"].replace("_", " ") for c in (r.get("overspending") or [])[:3]
            )
            if r.get("overspending") else ""
        )
        + "."
    ),
    "goal_allocation_result": lambda r: (
        f"Goal funding {r.get('status')}: surplus {_money(r.get('monthly_surplus'))}, "
        f"{_money(r.get('reserved_for_emergency'))} reserved for runway, "
        f"goals need {_money(r.get('total_required_monthly'))}/month"
        + (
            f" leaving a shortfall of {_money(r.get('shortfall'))}"
            if r.get("shortfall") else ""
        )
        + "."
    ),
    "scheme_matching_result": lambda r: (
        f"Government schemes: {r.get('eligible_count')} eligible worth about "
        f"{_money(r.get('estimated_annual_benefit'))}/year. Top matches: "
        + (
            ", ".join(
                f"{m.get('name')} ({m.get('match_score')}%)"
                for m in (r.get("matches") or [])[:3]
            )
            or "none"
        )
        + "."
    ),
    "eligibility_result": lambda r: (
        f"Scheme eligibility: {r.get('eligible_count')} of "
        f"{r.get('schemes_evaluated')} confirmed, "
        f"{r.get('possibly_eligible_count')} pending more information"
        + (
            " (need: " + ", ".join(
                f.replace("_", " ") for f in (r.get("ask_user_for") or [])
            ) + ")"
            if r.get("ask_user_for") else ""
        )
        + "."
    ),
    "bias_detection_result": lambda r: (
        f"Spending behaviour: {r.get('status')} over {r.get('months_analysed')} months, "
        f"costing about {_money(r.get('total_annual_cost'))}/year. "
        + (
            "Patterns: " + "; ".join(
                f["label"] for f in (r.get("findings") or [])[:3]
            ) + "."
            if r.get("findings") else "No material patterns."
        )
    ),
    "habit_formation_result": lambda r: (
        "Habit plan: "
        + (
            (r.get("keystone_habit") or {}).get("implementation_intention", "")
            + f" ({len(r.get('proposed_habits') or [])} habits proposed, worth about "
            f"{_money(r.get('total_potential_value'))}/year)."
            if r.get("keystone_habit") else "no habit changes indicated."
        )
    ),
    "nudge_strategy_result": lambda r: (
        f"Nudge programme: {r.get('active_count')} active, "
        f"{r.get('suppressed_count')} held back. Mechanisms: "
        f"{', '.join(r.get('mechanisms_used') or []) or 'none'}. "
        + "; ".join(
            f"{n['trigger']} -> {n['mechanism']}"
            for n in (r.get("active_nudges") or [])[:3]
        )
        + "."
    ),
    "literacy_result": lambda r: (
        f"Financial literacy {r.get('literacy_level')}: {r.get('gaps_identified')} gaps "
        f"worth about {_money(r.get('total_cost_of_gaps'))}/year. Topics to cover: "
        + (
            ", ".join(l["title"] for l in (r.get("curriculum") or [])[:3])
            or "none"
        )
        + "."
    ),
}


def summarise(result_key: str, result: dict[str, Any]) -> str:
    """One embeddable sentence describing an agent's finding."""
    summariser = SUMMARISERS.get(result_key)
    if summariser:
        try:
            return summariser(result)
        except Exception:
            pass                                    # fall through to the generic form
    agent = result_key.removesuffix("_result").replace("_", " ")
    return f"{agent} produced: {result}"[:600]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def memory_write_node(
    state: FinancialState,
    store: MemoryStore | None = None,
) -> dict[str, Any]:
    """
    Persist everything this run produced. The terminal node of every workflow.
    """
    store = store or get_store()
    user_id = state.get("user_id") or getattr(state.get("profile"), "user_id", "unknown")
    errors: list[str] = []
    written = 0

    for key in RESULT_KEYS:
        result = state.get(key)
        if not result:
            continue
        try:
            store.remember(
                user_id=user_id,
                memory_type=AGENT_MEMORY_TYPE.get(key, "episodic"),
                content=summarise(key, result),
                payload=result,
                source_agent=key.removesuffix("_result"),
            )
            written += 1
        except Exception as exc:
            errors.append(f"memory_write[{key}]: {exc!r}")

    # The judge's synthesis is the single most useful thing to recall later.
    if state.get("final_decision"):
        try:
            store.remember(
                user_id=user_id,
                memory_type="episodic",
                content=(
                    f"Asked: {state.get('query', '')}\n"
                    f"Decided: {state['final_decision'][:800]}"
                ),
                payload={"query": state.get("query", "")},
                source_agent="judge",
                importance=0.9,
            )
            written += 1
        except Exception as exc:
            errors.append(f"memory_write[final_decision]: {exc!r}")

    patch: dict[str, Any] = {"memory_written": written > 0}
    if errors:
        patch["errors"] = errors
    return patch


def memory_recall_node(
    state: FinancialState,
    store: MemoryStore | None = None,
    limit: int = 5,
    memory_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load relevant prior context. Runs at the FRONT of a workflow so agents can
    reason with history rather than treating every session as the first.
    """
    store = store or get_store()
    user_id = state.get("user_id") or getattr(state.get("profile"), "user_id", "unknown")
    query = state.get("query") or ""

    try:
        hits = (
            store.recall(user_id, query, limit=limit, memory_types=memory_types)
            if query
            else store.recent(user_id, limit=limit, memory_types=memory_types)
        )
    except Exception as exc:
        return {"recalled_memories": [], "errors": [f"memory_recall: {exc!r}"]}

    return {"recalled_memories": hits}
