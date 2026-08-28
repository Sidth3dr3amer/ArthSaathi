"""
Workflow endpoints.

`GET /workflows` lists the nine flows; `POST /workflow/{name}` runs one directly,
bypassing the router for when the UI already knows which flow the user picked
(a "Check a scheme" button, say).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ml.src.schemas.state import RESULT_KEYS, new_state
from ml.src.workflows.base import AGENT_ORDER
from ml.src.workflows.catalogue import WORKFLOWS, get_workflow

from ..deps import resolve_profile
from ..schemas import WorkflowRunRequest

router = APIRouter(tags=["workflows"])


@router.get("/workflows")
def list_workflows() -> dict[str, Any]:
    """Every workflow, with what it runs and whether it deliberates."""
    return {
        "count": len(WORKFLOWS),
        "workflows": [
            {
                "name": name,
                "label": spec["label"],
                "intent": spec["intent"],
                "description": spec["description"],
                "agents": list(spec["agents"]),
                "agent_count": len(spec["agents"]),
                "simulates": spec.get("simulate", False),
                "deliberates": spec.get("deliberate", False),
            }
            for name, spec in sorted(WORKFLOWS.items())
        ],
        "agents": list(AGENT_ORDER),
    }


@router.post("/workflow/{name}")
def run_workflow(name: str, request: WorkflowRunRequest) -> dict[str, Any]:
    """Run one named workflow directly."""
    if name not in WORKFLOWS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown workflow {name!r}; expected one of {sorted(WORKFLOWS)}",
        )

    profile, source = resolve_profile(request.user_id, request.profile)
    state = new_state(profile, request.query)
    if request.transactions is not None:
        state["transactions"] = request.transactions

    graph = get_workflow(name)
    try:
        final = graph.invoke(state)
        errors = list(final.get("errors", []))
    except Exception as exc:
        return {
            "workflow": name,
            "label": WORKFLOWS[name]["label"],
            "results": {},
            "errors": [f"workflow: {exc!r}"],
            "memory_written": False,
            "profile_source": source,
        }

    produced = [k for k in RESULT_KEYS if final.get(k)]
    return {
        "workflow": name,
        "label": WORKFLOWS[name]["label"],
        "steps": list(getattr(graph, "workflow_steps", [])),
        "agents_run": [k.removesuffix("_result") for k in produced],
        "results": {k: final[k] for k in produced},
        "allocation_plan": (final.get("utility_result") or {}).get("allocation_plan", []),
        "counterfactual": final.get("counterfactual_result"),
        "final_decision": final.get("final_decision", ""),
        "council_verdicts": [
            {"council": v.get("agent"), "stance": v.get("stance")}
            for v in final.get("verdicts", [])
        ],
        "recalled_memories": final.get("recalled_memories", []),
        "memory_written": bool(final.get("memory_written")),
        "profile_source": source,
        "errors": errors,
    }
