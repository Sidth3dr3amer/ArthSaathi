"""
Profile Agent -> Retriever + Context Builder.

The RAG half of the deck's slide-8 architecture: fetch the stored profile from
Postgres, similarity-search the pgvector memory store, and assemble a context
block for the LLM.

The design constraint is the context budget. Stuffing every memory into the
prompt is both expensive and counterproductive -- retrieval quality falls as
irrelevant material crowds out the relevant. So the builder:

  * keeps the profile summary (always relevant, small, and the thing most
    questions are actually about);
  * takes the top-k memories by similarity, then drops any below a floor,
    because a weak match is worse than no match -- it invites the model to
    connect things that are not connected;
  * hard-caps the assembled block by character budget, trimming the weakest
    matches first.
"""

from __future__ import annotations

from typing import Any

from ..memory.store import MemoryStore, get_store
from ..schemas.profile import UserProfile

#: Similarity below which a memory is noise rather than context.
SIMILARITY_FLOOR = 0.05

#: Character budget for the assembled memory section.
CONTEXT_BUDGET = 2_000


def retrieve(
    user_id: str,
    query: str,
    store: MemoryStore | None = None,
    limit: int = 8,
    memory_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fetch the profile and the memories most relevant to the query.

    Never raises: a store outage degrades to empty context so the assistant can
    still answer from the profile it was given.
    """
    store = store or get_store()
    errors: list[str] = []

    try:
        stored_profile = store.load_profile(user_id)
    except Exception as exc:
        stored_profile, _ = None, errors.append(f"profile: {exc!r}")

    try:
        memories = (
            store.recall(user_id, query, limit=limit, memory_types=memory_types)
            if query else store.recent(user_id, limit=limit, memory_types=memory_types)
        )
    except Exception as exc:
        memories, _ = [], errors.append(f"recall: {exc!r}")

    kept = [m for m in memories if m.get("similarity", 1.0) >= SIMILARITY_FLOOR]

    return {
        "profile": stored_profile,
        "memories": kept,
        "discarded_weak": len(memories) - len(kept),
        "errors": errors,
    }


def profile_summary(profile: UserProfile) -> str:
    """A compact, readable profile block. Only states what is actually known."""
    lines: list[str] = []
    if profile.name and profile.name != "Demo User":
        lines.append(f"Name: {profile.name}")
    if profile.age:
        lines.append(f"Age: {profile.age}")
    if profile.occupation:
        lines.append(f"Occupation: {profile.occupation} ({profile.job_type})")
    else:
        lines.append(f"Employment: {profile.job_type}")
    if profile.state:
        lines.append(f"State: {profile.state}")
    if profile.residence:
        lines.append(f"Area: {profile.residence}")
    if profile.dependents:
        lines.append(f"Dependants: {profile.dependents}")
    if profile.monthly_income:
        lines.append(f"Monthly income: Rs {profile.monthly_income:,.0f}")
    if profile.essential_expenses:
        lines.append(f"Essential expenses: Rs {profile.essential_expenses:,.0f}/month")
        lines.append(f"Monthly surplus: Rs {profile.monthly_surplus:,.0f}")
    if profile.existing_emergency_fund:
        lines.append(f"Emergency fund: Rs {profile.existing_emergency_fund:,.0f}")
    if profile.debts:
        lines.append(
            f"Debts: {len(profile.debts)} totalling Rs {profile.total_debt:,.0f} "
            f"(highest rate "
            f"{max((d.interest_rate or 0) for d in profile.debts):.0f}%)"
        )
    if profile.goals:
        lines.append("Goals: " + ", ".join(
            f"{g.name} (Rs {g.target_amount:,.0f} in {g.target_months} months)"
            for g in profile.goals
        ))
    lines.append(
        f"Health insurance: {'yes' if profile.has_health_insurance else 'no'}"
    )
    return "\n".join(lines)


def build_context(
    profile: UserProfile,
    retrieved: dict[str, Any],
    query: str = "",
    budget: int = CONTEXT_BUDGET,
) -> dict[str, Any]:
    """
    Assemble the prompt context, trimming the weakest memories to fit the budget.
    """
    summary = profile_summary(profile)
    memories = list(retrieved.get("memories", []))

    included: list[dict[str, Any]] = []
    lines: list[str] = []
    used = 0

    for index, memory in enumerate(memories):        # already ordered by similarity
        line = f"- [{memory['memory_type']}] {memory['content']}"

        if used + len(line) <= budget:
            included.append(memory)
            lines.append(line)
            used += len(line)
            continue

        # The single best match is always included, truncated if it has to be.
        # Returning no context at all because the budget is tight is a worse
        # failure than returning the most relevant memory in shortened form.
        if index == 0:
            truncated = line[: max(budget, 80)].rstrip() + "..."
            included.append(memory)
            lines.append(truncated)
            used += len(truncated)

    memory_block = "\n".join(lines) or "(nothing recorded from previous conversations)"

    context = (
        f"WHAT WE KNOW ABOUT THIS USER\n{summary}\n\n"
        f"RELEVANT HISTORY\n{memory_block}"
    )

    return {
        "context": context,
        "profile_summary": summary,
        "memories_included": len(included),
        "memories_dropped": len(memories) - len(included),
        "characters": len(context),
        "budget": budget,
    }


def retrieve_and_build(
    profile: UserProfile,
    query: str,
    store: MemoryStore | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Retriever and Context Builder in one call."""
    retrieved = retrieve(profile.user_id, query, store=store, limit=limit)
    built = build_context(profile, retrieved, query)
    return {**built, "retrieval": retrieved}
