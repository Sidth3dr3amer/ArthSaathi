"""
The main conversational entry point.

`POST /chat` is the whole system in one call: route the message, run the right
workflow, deliberate if the question warrants it, remember the outcome.

`POST /ask` is the cheaper read path -- it answers from retrieved profile and
memory without running any council, which is what a follow-up question like
"what did we decide?" actually needs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ml.src.profile_agent.response_gen import answer_with_context
from ml.src.workflows.orchestrator import run, summarise_run

from ..deps import resolve_profile
from ..schemas import AskRequest, ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

#: Longest recommendation line we will return. Anything longer is a payload that
#: escaped into a summary field, not a sentence for a human.
MAX_RECOMMENDATION_CHARS = 400


def _as_line(item: Any) -> str | None:
    """
    Coerce one recommendation entry to a display line.

    `summarise_run` now normalises structured entries upstream (the credit-card
    agent returns full card objects, which previously leaked ~15 KB of card
    database into a response documented as a "compact view").

    This layer is kept deliberately as defence in depth, and because it does one
    thing upstream does not: it caps line length, so a verbose future agent
    cannot bloat the endpoint's `dict[str, list[str]]` contract.
    """
    if isinstance(item, str):
        return item[:MAX_RECOMMENDATION_CHARS]

    if isinstance(item, dict):
        card = item.get("card")
        if isinstance(card, dict) and card.get("card_name"):
            value = item.get("net_value")
            suffix = f" — net annual value Rs {value:,.0f}" if isinstance(value, (int, float)) else ""
            return f"{card['card_name']}{suffix}"
        for key in ("label", "name", "title", "message", "recommendation"):
            if isinstance(item.get(key), str):
                return item[key][:MAX_RECOMMENDATION_CHARS]
        return None

    return None


def normalise_recommendations(raw: dict[str, Any]) -> dict[str, list[str]]:
    """Reduce every agent's recommendations to display lines, dropping the rest."""
    cleaned: dict[str, list[str]] = {}
    for agent, items in (raw or {}).items():
        if not isinstance(items, list):
            continue
        lines = [line for line in (_as_line(i) for i in items) if line]
        if lines:
            cleaned[agent] = lines
    return cleaned


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    """
    Run a user message through the full orchestrator.

    A failure inside the agent layer is reported in `errors` with whatever the
    run did manage to produce, rather than surfacing as a 500 -- a user asking
    about their finances should never get a stack trace because one network-bound
    agent timed out.
    """
    profile, source = resolve_profile(request.user_id, request.profile)

    try:
        result = run(
            request.message,
            profile,
            use_llm_router=request.use_llm_router,
            transactions=request.transactions,
        )
        summary = summarise_run(result)
    except Exception as exc:
        return ChatResponse(
            intent="general",
            workflow="unavailable",
            agents_run=[],
            recommendations={},
            allocation_plan=[],
            final_decision="",
            council_verdicts=[],
            memory_written=False,
            errors=[f"orchestrator: {exc!r}"],
            elapsed_seconds=0.0,
        ).model_dump()

    summary["recommendations"] = normalise_recommendations(summary.get("recommendations", {}))
    summary["profile_source"] = source
    return summary


@router.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Answer from stored profile and memory only. No councils, no simulation."""
    profile, source = resolve_profile(request.user_id, request.profile)

    try:
        answer = answer_with_context(request.question, profile)
    except Exception as exc:
        return {
            "answer": "",
            "method": f"error: {exc!r}",
            "memories_included": 0,
            "profile_source": source,
            "errors": [repr(exc)],
        }

    return {
        "answer": answer["answer"],
        "method": answer["method"],
        "language": answer["language"],
        "memories_included": answer["memories_included"],
        "profile_source": source,
        "errors": [],
    }
