"""
Shared FastAPI dependencies.

Two jobs: pick the memory store the app runs against, and resolve a request's
user into a `UserProfile`.

Profile resolution deserves a note. A request may carry a full profile inline
(the frontend holding state) or just a `user_id` (a returning user whose profile
lives in Postgres). Supporting both is what lets the same endpoint serve the
onboarding flow and a repeat session. An inline profile always wins over the
stored one, because it is the newer information -- but its `user_id` is forced
to match the request so a client cannot write into another user's record.
"""

from __future__ import annotations

from typing import Any

from ml.src.common import config
from ml.src.memory.store import (
    InMemoryStore,
    MemoryStore,
    PostgresMemoryStore,
    get_store,
    set_store,
)
from ml.src.profile_agent.updater import load_profile
from ml.src.schemas.profile import UserProfile


def init_store() -> MemoryStore:
    """
    Choose the store at startup.

    Postgres when `DATABASE_URL` is configured and reachable, otherwise the
    in-memory store. A database that is configured but unreachable must not stop
    the API from serving -- the agents all work without memory, just without
    continuity, so degrading is strictly better than refusing to start.
    """
    if not config.DATABASE_URL:
        store = InMemoryStore()
        set_store(store)
        return store

    try:
        store = PostgresMemoryStore()
        store.create_schema()
    except Exception:
        store = InMemoryStore()

    set_store(store)
    return store


def store_kind(store: MemoryStore | None = None) -> str:
    return type(store or get_store()).__name__


def resolve_profile(
    user_id: str,
    inline: dict[str, Any] | UserProfile | None = None,
    store: MemoryStore | None = None,
) -> tuple[UserProfile, str]:
    """
    Resolve a request to a profile.

    Returns `(profile, source)` where source is one of `inline`, `stored`, or
    `new`, so a caller can tell whether it is talking to a known user.
    """
    if inline is not None:
        profile = (
            inline if isinstance(inline, UserProfile)
            else UserProfile.model_validate(inline)
        )
        # The path/body user_id is authoritative: a client must not be able to
        # write into someone else's record by supplying a different id inline.
        if profile.user_id != user_id:
            profile = profile.model_copy(update={"user_id": user_id})
        return profile, "inline"

    stored = load_profile(user_id, store or get_store())
    if stored is not None:
        return stored, "stored"

    return UserProfile(user_id=user_id), "new"
