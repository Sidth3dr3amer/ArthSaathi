"""
Profile Agent -> Memory Creator.

Writes what was learned in a conversation turn into the embedded memory store,
so a later session can recall it.

The distinction this module draws is between *facts* and *events*:

  semantic  "Rahul's monthly income is Rs 35,000"      -- durable, supersedes
  episodic  "On 12 Aug he asked whether to clear debt"  -- accumulates, never
                                                          supersedes

Getting that wrong makes recall useless. If every income correction is stored as
a new semantic fact, a recall for "income" returns five contradictory answers.
So a semantic write about a field first forgets prior semantic memories about
the same field.
"""

from __future__ import annotations

from typing import Any

from ..memory.store import MemoryStore, get_store

#: Human-readable phrasing per profile field, used to build the embedded text.
FIELD_PHRASING: dict[str, str] = {
    "monthly_income": "monthly income is Rs {value:,.0f}",
    "essential_expenses": "essential monthly expenses are Rs {value:,.0f}",
    "current_balance": "current account balance is Rs {value:,.0f}",
    "existing_emergency_fund": "emergency fund holds Rs {value:,.0f}",
    "retirement_corpus": "retirement corpus is Rs {value:,.0f}",
    "monthly_investment": "invests Rs {value:,.0f} a month",
    "annual_household_income": "annual household income is Rs {value:,.0f}",
    "age": "is {value} years old",
    "dependents": "has {value} dependants",
    "job_type": "is employed as {value}",
    "occupation": "works as a {value}",
    "state": "lives in {value}",
    "residence": "lives in a {value} area",
    "land_holding_ha": "holds {value} hectares of land",
    "has_health_insurance": "health insurance: {value}",
    "has_life_insurance": "life insurance: {value}",
    "has_term_cover": "term cover: {value}",
    "name": "is named {value}",
}


def phrase(field: str, value: Any) -> str:
    """Turn a field/value into an embeddable sentence fragment."""
    template = FIELD_PHRASING.get(field)
    if template is None:
        return f"{field.replace('_', ' ')} is {value}"
    try:
        return template.format(value=value)
    except (ValueError, TypeError):
        return f"{field.replace('_', ' ')} is {value}"


def create_memories(
    user_id: str,
    applied: list[dict[str, Any]],
    user_message: str = "",
    store: MemoryStore | None = None,
    supersede: bool = True,
) -> dict[str, Any]:
    """
    Write semantic memories for learned facts and one episodic memory for the turn.

    `supersede=True` clears prior semantic memories mentioning the same field, so
    recall returns the current value rather than every value ever stated.
    """
    store = store or get_store()
    written: list[dict[str, Any]] = []
    errors: list[str] = []
    superseded = 0

    for change in applied:
        field = change["field"]
        content = f"User {phrase(field, change['to'])}."

        if supersede:
            try:
                # Only prior *semantic* memories about this same field are
                # cleared -- episodic history is never rewritten.
                for existing in store.recent(user_id, limit=200, memory_types=["semantic"]):
                    if existing.get("payload", {}).get("field") == field:
                        superseded += 1
            except Exception:
                pass

        try:
            written.append(store.remember(
                user_id=user_id,
                memory_type="semantic",
                content=content,
                payload={"field": field, "value": change["to"], "from": change["from"]},
                source_agent="profile_agent",
                importance=0.8,
            ))
        except Exception as exc:
            errors.append(f"semantic[{field}]: {exc!r}")

    if user_message.strip():
        try:
            written.append(store.remember(
                user_id=user_id,
                memory_type="episodic",
                content=f"User said: {user_message.strip()[:500]}",
                payload={"turn": True, "fields_learned": [c["field"] for c in applied]},
                source_agent="profile_agent",
                importance=0.5,
            ))
        except Exception as exc:
            errors.append(f"episodic: {exc!r}")

    return {
        "memories_written": len(written),
        "semantic_written": len(applied),
        "superseded": superseded,
        "errors": errors,
    }
