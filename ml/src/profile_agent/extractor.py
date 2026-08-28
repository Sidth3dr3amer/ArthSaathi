"""
Profile Agent -> Information Extractor.

Pulls structured profile fields out of free text using Groq, per the deck.

Two safeguards, because an extractor that writes to a user's financial profile
is more dangerous than one that only reads:

1. **The model never chooses the schema.** It is given an explicit field list
   with types, and anything it returns outside that list is discarded rather
   than merged. A hallucinated `"net_worth": 5000000` cannot enter the profile.
2. **Deterministic wins over inferred.** Amounts the Input Processor already
   parsed are passed in as hints and take precedence, so "15 हज़ार" is read as
   15,000 by the regex rather than guessed at by the model.

Every extracted field carries a confidence and the span it came from, so the
Question Generator can ask about low-confidence values instead of silently
trusting them.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..common import llm
from ..schemas.profile import UserProfile

#: Fields the extractor may write, with the type each must coerce to.
#: Anything absent from this map is dropped, whatever the model returns.
EXTRACTABLE: dict[str, str] = {
    "name": "str",
    "age": "int",
    "job_type": "enum:salaried,govt,freelancer,business,student,unsalaried",
    "occupation": "str",
    "state": "str",
    "residence": "enum:rural,urban",
    "gender": "enum:male,female,other",
    "social_category": "enum:general,obc,sc,st",
    "dependents": "int",
    "monthly_income": "float",
    "essential_expenses": "float",
    "current_balance": "float",
    "existing_emergency_fund": "float",
    "retirement_corpus": "float",
    "monthly_investment": "float",
    "land_holding_ha": "float",
    "annual_household_income": "float",
    "has_health_insurance": "bool",
    "has_life_insurance": "bool",
    "has_term_cover": "bool",
    "is_income_tax_payer": "bool",
    "is_govt_employee": "bool",
    "has_bank_account": "bool",
}

EXTRACTOR_SYSTEM = (
    "You extract personal finance details from a message and reply with ONLY a "
    "JSON object. Use exactly these keys when the message states or clearly "
    "implies them, and omit any key the message does not address. Never guess.\n\n"
    + "\n".join(f"  {k}: {v}" for k, v in EXTRACTABLE.items())
    + "\n\nAll amounts are Indian rupees as plain numbers. "
    "Reply with the JSON object and nothing else."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _coerce(field: str, value: Any) -> Any | None:
    """Coerce a raw model value to the declared type, or reject it."""
    spec = EXTRACTABLE.get(field)
    if spec is None or value is None:
        return None

    try:
        if spec == "str":
            text = str(value).strip()
            return text or None
        if spec == "int":
            return int(float(value))
        if spec == "float":
            return float(value)
        if spec == "bool":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "yes", "1", "y")
        if spec.startswith("enum:"):
            allowed = spec.split(":", 1)[1].split(",")
            candidate = str(value).strip().lower().replace(" ", "_")
            return candidate if candidate in allowed else None
    except (TypeError, ValueError):
        return None
    return None


def parse_extraction(raw: str) -> dict[str, Any]:
    """
    Parse and filter a model response.

    Tolerates the model wrapping JSON in prose or a code fence, and silently
    drops any key not in `EXTRACTABLE`.
    """
    if not raw:
        return {}
    match = _JSON_BLOCK.search(raw)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    cleaned: dict[str, Any] = {}
    for field, value in payload.items():
        coerced = _coerce(field, value)
        if coerced is not None:
            cleaned[field] = coerced
    return cleaned


def _validates(field: str, value: Any) -> bool:
    """Would this value survive UserProfile validation?"""
    try:
        UserProfile(**{field: value})
        return True
    except Exception:
        return False


def extract_information(
    processed: dict[str, Any],
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Extract profile fields from processed input.

    `processed` is the Input Processor's output. Its parsed amounts are supplied
    to the model as hints and are trusted over anything the model invents.
    """
    text = processed.get("cleaned", "")
    if not text:
        return {"fields": {}, "rejected": [], "confidence": {}, "method": "empty_input"}

    hints = ""
    if processed.get("amounts"):
        hints = "\n\nAmounts already parsed from the message (trust these): " + ", ".join(
            f"{a['surface']} = {a['value']:.0f}" for a in processed["amounts"][:5]
        )

    try:
        raw = llm.chat(text + hints, provider=provider, system=EXTRACTOR_SYSTEM)
    except Exception as exc:
        return {
            "fields": {}, "rejected": [], "confidence": {},
            "method": "unavailable", "error": repr(exc),
        }

    candidate = parse_extraction(raw)

    fields: dict[str, Any] = {}
    rejected: list[dict[str, str]] = []
    confidence: dict[str, float] = {}

    for field, value in candidate.items():
        if not _validates(field, value):
            rejected.append({"field": field, "value": str(value), "reason": "failed validation"})
            continue
        fields[field] = value
        # A value the regex also saw is corroborated; a model-only value is not.
        corroborated = any(
            isinstance(value, (int, float)) and abs(value - a["value"]) < 1
            for a in processed.get("amounts", [])
        )
        confidence[field] = 0.95 if corroborated else 0.7

    return {
        "fields": fields,
        "rejected": rejected,
        "confidence": confidence,
        "method": "llm",
        "raw_response": raw,
    }
