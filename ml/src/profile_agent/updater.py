"""
Profile Agent -> Profile Updater.

Merges extracted fields into the stored profile and persists it to Postgres.

The merge is conservative on purpose. A profile is long-lived and a bad write is
expensive to notice: if the extractor mishears "50" as monthly income, every
downstream council silently produces nonsense. So:

  * a low-confidence value never overwrites an existing high-confidence one;
  * a change large enough to be suspicious is flagged for confirmation rather
    than applied silently;
  * every write records what changed, so the response generator can say
    "I've noted your income as X" and give the user a chance to correct it.
"""

from __future__ import annotations

from typing import Any

from ..memory.store import MemoryStore, get_store
from ..schemas.profile import UserProfile

#: Below this, a value never overwrites an existing one.
OVERWRITE_CONFIDENCE = 0.8

#: A numeric change larger than this multiple is surfaced for confirmation.
SUSPICIOUS_CHANGE_FACTOR = 3.0

#: Fields where a wrong value does the most damage downstream.
CRITICAL_FIELDS = {
    "monthly_income", "essential_expenses", "age", "dependents",
    "existing_emergency_fund", "retirement_corpus",
}


def merge_fields(
    profile: UserProfile,
    fields: dict[str, Any],
    confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Merge extracted fields into a profile copy.

    Returns the updated profile plus an audit of what was applied, skipped, or
    held back for confirmation. Never mutates the input.
    """
    confidence = confidence or {}
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    needs_confirmation: list[dict[str, Any]] = []
    update: dict[str, Any] = {}

    defaults = UserProfile()

    for field, new_value in fields.items():
        if not hasattr(profile, field):
            skipped.append({"field": field, "reason": "unknown field"})
            continue

        current = getattr(profile, field)
        default = getattr(defaults, field, None)
        is_unset = current == default or current in (None, 0, "")
        score = confidence.get(field, 0.7)

        if current == new_value:
            skipped.append({"field": field, "reason": "unchanged"})
            continue

        # An established value needs real confidence to be overwritten.
        if not is_unset and score < OVERWRITE_CONFIDENCE:
            needs_confirmation.append({
                "field": field, "current": current, "proposed": new_value,
                "reason": f"low confidence ({score:.2f}) against an existing value",
            })
            continue

        # A large jump on a critical field is surfaced, not applied silently.
        if (
            field in CRITICAL_FIELDS
            and not is_unset
            and isinstance(current, (int, float))
            and isinstance(new_value, (int, float))
            and current > 0
            and (new_value / current > SUSPICIOUS_CHANGE_FACTOR
                 or current / max(new_value, 1e-9) > SUSPICIOUS_CHANGE_FACTOR)
        ):
            needs_confirmation.append({
                "field": field, "current": current, "proposed": new_value,
                "reason": f"changes by more than {SUSPICIOUS_CHANGE_FACTOR:.0f}x",
            })
            continue

        update[field] = new_value
        applied.append({"field": field, "from": current, "to": new_value,
                        "confidence": score})

    updated = profile.model_copy(update=update) if update else profile

    return {
        "profile": updated,
        "applied": applied,
        "skipped": skipped,
        "needs_confirmation": needs_confirmation,
        "changed": bool(update),
    }


def save_profile(
    profile: UserProfile, store: MemoryStore | None = None
) -> dict[str, Any]:
    """Persist the profile document. Never raises into a conversation."""
    store = store or get_store()
    try:
        store.save_profile(profile.user_id, profile.model_dump(mode="json"))
        return {"saved": True}
    except Exception as exc:
        return {"saved": False, "error": repr(exc)}


def load_profile(
    user_id: str, store: MemoryStore | None = None
) -> UserProfile | None:
    """Load a stored profile, or None when the user is new or the store is down."""
    store = store or get_store()
    try:
        stored = store.load_profile(user_id)
    except Exception:
        return None
    if not stored:
        return None
    try:
        return UserProfile.model_validate(stored)
    except Exception:
        return None


def update_profile(
    profile: UserProfile,
    extraction: dict[str, Any],
    store: MemoryStore | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Merge an extraction into a profile and persist the result."""
    merged = merge_fields(
        profile, extraction.get("fields", {}), extraction.get("confidence", {})
    )
    if persist and merged["changed"]:
        merged.update(save_profile(merged["profile"], store))
    else:
        merged["saved"] = False
    return merged
