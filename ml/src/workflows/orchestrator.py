"""
The Decision Orchestrator.

Top-level entry point: a user message and a profile go in, a routed, executed,
deliberated and remembered result comes out. This is the whole
`ArthaSaathi Workflow Orchestration` diagram from the deck, in one call.

    User Query -> Router -> Workflow -> Agents -> [Simulation]
               -> [Deliberation] -> Recommendation -> Memory Update
"""

from __future__ import annotations

import time
from typing import Any

from ..common import llm
from ..decision.router import route
from ..memory.store import MemoryStore, set_store
from ..schemas.profile import UserProfile
from ..schemas.state import RESULT_KEYS, FinancialState, new_state
from .catalogue import WORKFLOWS, get_workflow, workflow_for_intent


def run(
    query: str,
    profile: UserProfile,
    *,
    store: MemoryStore | None = None,
    use_llm_router: bool = True,
    transactions: list[dict[str, Any]] | None = None,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Route a query to the right workflow and run it end to end.

    Returns the final state plus a trace of what ran, so the API and the UI can
    show the reasoning path rather than only the answer.
    """
    if store is not None:
        set_store(store)

    started = time.time()
    decision = route(query, use_llm=use_llm_router, provider=provider)
    workflow_name = workflow_for_intent(decision["intent"])
    spec = WORKFLOWS[workflow_name]

    state: FinancialState = new_state(profile, query)
    if transactions is not None:
        state["transactions"] = transactions
    state["intent"] = decision["intent"]
    state["routed_councils"] = decision["councils"]
    state["routing"] = decision

    graph = get_workflow(workflow_name)
    try:
        final = graph.invoke(state)
        error = None
    except Exception as exc:
        final, error = state, repr(exc)

    produced = [k for k in RESULT_KEYS if final.get(k)]

    return {
        "state": final,
        "intent": decision["intent"],
        "workflow": workflow_name,
        "workflow_label": spec["label"],
        "routing": decision,
        "steps": list(getattr(graph, "workflow_steps", [])),
        "agents_run": produced,
        "agent_count": len(produced),
        "deliberated": bool(final.get("verdicts")),
        "simulated": bool(final.get("counterfactual_result")),
        "memory_written": bool(final.get("memory_written")),
        "recalled": len(final.get("recalled_memories", [])),
        "errors": final.get("errors", []) + ([error] if error else []),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def summarise_run(result: dict[str, Any]) -> dict[str, Any]:
    """
    A compact, JSON-safe view of a run for an API response or a dashboard.

    Deliberately excludes the raw profile and full agent payloads -- those are
    available on `result["state"]` when a caller wants them.
    """
    state = result["state"]

    def _as_lines(value: Any) -> list[str]:
        """
        Coerce an agent's `recommendations` to a list of strings.

        Most agents already return strings, but the credit-card agent returns
        full card objects -- which turned this "compact" summary into a 15 KB
        dump of the card database. Normalising here keeps the contract
        (`dict[str, list[str]]`) true for every agent, present and future.
        """
        if not isinstance(value, list):
            return []
        lines: list[str] = []
        for item in value:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                card = item.get("card") if isinstance(item.get("card"), dict) else None
                name = (
                    (card or {}).get("card_name")
                    or item.get("name")
                    or item.get("label")
                    or item.get("title")
                )
                amount = item.get("net_value", item.get("annual_value"))
                if name and isinstance(amount, (int, float)):
                    lines.append(f"{name} — net Rs {amount:,.0f}/year")
                elif name:
                    lines.append(str(name))
            else:
                lines.append(str(item))
        return lines

    return {
        "intent": result["intent"],
        "workflow": result["workflow_label"],
        "agents_run": result["agents_run"],
        "recommendations": {
            key.removesuffix("_result"): lines
            for key in result["agents_run"]
            if isinstance(state.get(key), dict)
            and (lines := _as_lines(state[key].get("recommendations")))
        },
        "allocation_plan": (state.get("utility_result") or {}).get("allocation_plan", []),
        "final_decision": state.get("final_decision", ""),
        "council_verdicts": [
            {"council": v.get("agent"), "stance": v.get("stance")}
            for v in state.get("verdicts", [])
        ],
        "memory_written": result["memory_written"],
        "errors": result["errors"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
