"""
Memory endpoints.

Read and search a user's memories, and delete them. The delete route exists
because "forget me" is a reasonable thing for a user to ask of a system that
remembers their finances.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ml.src.memory.models import MEMORY_TYPES
from ml.src.memory.store import get_store

from ..schemas import RecallRequest

router = APIRouter(prefix="/memory", tags=["memory"])


def _validate_types(memory_types: list[str] | None) -> list[str] | None:
    if not memory_types:
        return None
    unknown = [t for t in memory_types if t not in MEMORY_TYPES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown memory types {unknown}; expected from {list(MEMORY_TYPES)}",
        )
    return memory_types


@router.get("/types")
def memory_types() -> dict[str, Any]:
    return {"types": list(MEMORY_TYPES)}


@router.get("/{user_id}")
def recent_memories(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    memory_type: str | None = None,
) -> dict[str, Any]:
    """Most recent memories, newest first."""
    types = _validate_types([memory_type] if memory_type else None)
    try:
        memories = get_store().recent(user_id, limit=limit, memory_types=types)
    except Exception as exc:
        return {"user_id": user_id, "memories": [], "count": 0, "errors": [repr(exc)]}

    return {
        "user_id": user_id,
        "memories": memories,
        "count": len(memories),
        "errors": [],
    }


@router.post("/{user_id}/recall")
def recall_memories(user_id: str, request: RecallRequest) -> dict[str, Any]:
    """Semantic search over a user's memories."""
    types = _validate_types(request.memory_types)
    try:
        memories = get_store().recall(
            user_id, request.query, limit=request.limit, memory_types=types
        )
    except Exception as exc:
        return {
            "user_id": user_id, "query": request.query,
            "memories": [], "count": 0, "errors": [repr(exc)],
        }

    return {
        "user_id": user_id,
        "query": request.query,
        "memories": memories,
        "count": len(memories),
        "errors": [],
    }


@router.delete("/{user_id}")
def forget(user_id: str, memory_type: str | None = None) -> dict[str, Any]:
    """Delete a user's memories, optionally only one type."""
    _validate_types([memory_type] if memory_type else None)
    try:
        removed = get_store().forget(user_id, memory_type)
    except Exception as exc:
        return {"user_id": user_id, "removed": 0, "errors": [repr(exc)]}

    return {"user_id": user_id, "removed": removed, "errors": []}
