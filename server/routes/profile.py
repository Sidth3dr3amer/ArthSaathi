"""
Profile Agent endpoints.

`POST /profile/turn` drives the deck's "Teaching Saathis" onboarding: one message
in, an updated profile and the next question out. `GET /profile/{id}/questions`
backs the "3 of 7 questions, 45% complete" progress screen.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ml.src.memory.store import get_store
from ml.src.profile_agent import question_gen
from ml.src.profile_agent.response_gen import run_profile_agent
from ml.src.profile_agent.updater import load_profile, save_profile
from ml.src.schemas.profile import UserProfile

from ..deps import resolve_profile
from ..schemas import ProfileTurnRequest, ProfileTurnResponse

router = APIRouter(prefix="/profile", tags=["profile"])


@router.post("/turn", response_model=ProfileTurnResponse)
def profile_turn(request: ProfileTurnRequest) -> dict[str, Any]:
    """One onboarding conversation turn."""
    profile, _ = resolve_profile(request.user_id, request.profile)

    try:
        result = run_profile_agent(
            request.message, profile, store=get_store(), persist=request.persist
        )
    except Exception as exc:
        # Degrade to "we heard you but learned nothing" rather than a 500.
        plan = question_gen.question_plan(profile)
        return ProfileTurnResponse(
            user_id=request.user_id,
            response="Sorry — I could not process that just now. Could you try again?",
            profile=profile.model_dump(mode="json"),
            next_question=plan["next_question"],
            completeness=plan["completeness"],
            needs_confirmation=[],
            stages={"error": repr(exc)},
        ).model_dump()

    return {
        "user_id": request.user_id,
        "response": result["response"],
        "profile": result["profile"].model_dump(mode="json"),
        "next_question": result["next_question"],
        "completeness": result["completeness"],
        "needs_confirmation": result["needs_confirmation"],
        # The raw extractor payload is large and includes the model's own text;
        # the stage summary is what a UI needs to show progress.
        "stages": {
            "language": result["stages"]["input_processor"]["language"],
            "amounts_found": len(result["stages"]["input_processor"]["amounts"]),
            "fields_extracted": list(result["stages"]["extractor"]["fields"]),
            "fields_applied": [c["field"] for c in result["stages"]["updater"]["applied"]],
            "fields_rejected": [r["field"] for r in result["stages"]["extractor"]["rejected"]],
            "memories_written": result["stages"]["memory_creator"]["memories_written"],
        },
    }


@router.get("/{user_id}")
def get_profile(user_id: str) -> dict[str, Any]:
    """Load a stored profile. 404 when the user has never been seen."""
    profile = load_profile(user_id, get_store())
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no profile for user {user_id!r}")

    plan = question_gen.question_plan(profile)
    return {
        "profile": profile.model_dump(mode="json"),
        "completeness": plan["completeness"],
        "can_advise": plan["completeness"]["can_advise"],
    }


@router.put("/{user_id}")
def put_profile(user_id: str, profile: UserProfile) -> dict[str, Any]:
    """Replace a stored profile outright (the frontend's save button)."""
    if profile.user_id != user_id:
        profile = profile.model_copy(update={"user_id": user_id})

    outcome = save_profile(profile, get_store())
    if not outcome["saved"]:
        # The write failed but the caller's data is intact -- report it plainly.
        return {
            "saved": False,
            "errors": [outcome.get("error", "unknown store failure")],
            "profile": profile.model_dump(mode="json"),
        }

    return {"saved": True, "errors": [], "profile": profile.model_dump(mode="json")}


@router.get("/{user_id}/questions")
def get_questions(user_id: str, limit: int = 7) -> dict[str, Any]:
    """
    The onboarding queue.

    Works for an unknown user too -- a new user is exactly who needs the
    question list, so this returns the full queue rather than a 404.
    """
    profile = load_profile(user_id, get_store()) or UserProfile(user_id=user_id)
    plan = question_gen.question_plan(profile, limit=limit)
    return {
        "user_id": user_id,
        "next_question": plan["next_question"],
        "queue": plan["queue"],
        "remaining": plan["remaining"],
        "completeness": plan["completeness"],
        "blocked_councils": plan["blocked_councils"],
    }
