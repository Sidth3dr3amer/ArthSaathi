"""
Profile Agent -> Question Generator.

Decides what to ask next. This is the engine behind the deck's "Teaching
Saathis" onboarding screen ("3 of 7 questions, 45% complete").

The ordering is the whole design. Asking seven questions in schema order gets
abandoned; asking the two that unlock the most analysis does not. So each field
carries:

  unlocks   which councils cannot run without it
  weight    how much analysis it enables
  phrasing  a question a person would actually answer, in plain language

and the agent asks in descending value, one at a time, stopping as soon as the
profile is good enough to produce a real answer.
"""

from __future__ import annotations

from typing import Any

from ..schemas.profile import UserProfile

#: field -> (question, why we ask, councils unlocked, weight)
QUESTION_BANK: dict[str, dict[str, Any]] = {
    "monthly_income": {
        "question": "Roughly how much do you earn in a month?",
        "why": "Almost every recommendation is sized against your income.",
        "unlocks": ["cashflow", "risk", "growth", "benefits"],
        "weight": 10,
        "follow_up": "An approximate figure is fine — it does not need to be exact.",
    },
    "essential_expenses": {
        "question": "About how much do you spend on essentials each month — rent, food, bills?",
        "why": "This sets your emergency fund target and how much you can save.",
        "unlocks": ["cashflow", "risk"],
        "weight": 9,
    },
    "dependents": {
        "question": "How many people depend on your income?",
        "why": "It changes how much cover and runway you need.",
        "unlocks": ["risk"],
        "weight": 7,
    },
    "age": {
        "question": "How old are you?",
        "why": "It drives retirement planning and how much risk your portfolio can take.",
        "unlocks": ["growth"],
        "weight": 7,
    },
    "job_type": {
        "question": "Are you salaried, self-employed, a business owner, or something else?",
        "why": "Income stability changes how large a safety net you need.",
        "unlocks": ["risk", "growth", "benefits"],
        "weight": 6,
    },
    "has_health_insurance": {
        "question": "Do you have health insurance right now?",
        "why": "An uninsured hospital bill is the most common cause of sudden debt.",
        "unlocks": ["risk"],
        "weight": 6,
    },
    "existing_emergency_fund": {
        "question": "How much do you have saved that you could reach in an emergency?",
        "why": "It tells us how many months you could cover if income stopped.",
        "unlocks": ["risk", "cashflow"],
        "weight": 6,
    },
    "occupation": {
        "question": "What kind of work do you do?",
        "why": "Several government schemes are tied to occupation.",
        "unlocks": ["benefits"],
        "weight": 5,
    },
    "residence": {
        "question": "Do you live in a rural or urban area?",
        "why": "Housing and employment schemes differ between the two.",
        "unlocks": ["benefits"],
        "weight": 4,
    },
    "state": {
        "question": "Which state do you live in?",
        "why": "State schemes come on top of the central ones.",
        "unlocks": ["benefits"],
        "weight": 3,
    },
    "land_holding_ha": {
        "question": "Do you own any agricultural land, and roughly how much?",
        "why": "PM-KISAN and crop insurance depend on landholding size.",
        "unlocks": ["benefits"],
        "weight": 3,
    },
    "retirement_corpus": {
        "question": "Have you saved anything specifically for retirement — EPF, NPS, PPF?",
        "why": "It sets the starting point for your retirement projection.",
        "unlocks": ["growth"],
        "weight": 3,
    },
}

#: Without these, no council can say anything useful.
ESSENTIAL_FIELDS = ("monthly_income", "essential_expenses")


def _is_unset(profile: UserProfile, field: str) -> bool:
    default = getattr(UserProfile(), field, None)
    value = getattr(profile, field, None)
    if field == "has_health_insurance":
        return False          # boolean default is a real answer, never re-asked
    return value == default or value in (None, 0, "")


def missing_fields(profile: UserProfile) -> list[str]:
    """Fields in the bank that this profile has not answered, most valuable first."""
    missing = [f for f in QUESTION_BANK if _is_unset(profile, f)]
    return sorted(missing, key=lambda f: QUESTION_BANK[f]["weight"], reverse=True)


def completeness(profile: UserProfile) -> dict[str, Any]:
    """How much of the question bank is answered, by count and by weight."""
    total_weight = sum(q["weight"] for q in QUESTION_BANK.values())
    missing = missing_fields(profile)
    missing_weight = sum(QUESTION_BANK[f]["weight"] for f in missing)
    answered = len(QUESTION_BANK) - len(missing)

    return {
        "answered": answered,
        "total": len(QUESTION_BANK),
        "percent": round((answered / len(QUESTION_BANK)) * 100, 1),
        "weighted_percent": round(
            ((total_weight - missing_weight) / total_weight) * 100, 1
        ),
        "missing": missing,
        "can_advise": all(not _is_unset(profile, f) for f in ESSENTIAL_FIELDS),
    }


def generate_question(profile: UserProfile) -> dict[str, Any] | None:
    """
    The single most valuable question to ask next, or None when ready.

    Stops once the essentials are known and nothing high-value remains, rather
    than marching through the whole bank.
    """
    state = completeness(profile)
    if not state["missing"]:
        return None

    field = state["missing"][0]
    spec = QUESTION_BANK[field]

    # Once we can advise, only keep asking if the remaining question is valuable.
    if state["can_advise"] and spec["weight"] < 5:
        return None

    return {
        "field": field,
        "question": spec["question"],
        "why_we_ask": spec["why"],
        "follow_up": spec.get("follow_up"),
        "unlocks": spec["unlocks"],
        "weight": spec["weight"],
        "progress": {
            "answered": state["answered"],
            "total": state["total"],
            "percent": state["percent"],
        },
    }


def question_plan(profile: UserProfile, limit: int = 7) -> dict[str, Any]:
    """
    The onboarding sequence: the next few questions in priority order.

    Mirrors the deck's "3 of 7 questions" progress screen.
    """
    state = completeness(profile)
    queue = [
        {
            "field": field,
            "question": QUESTION_BANK[field]["question"],
            "why_we_ask": QUESTION_BANK[field]["why"],
            "unlocks": QUESTION_BANK[field]["unlocks"],
        }
        for field in state["missing"][:limit]
    ]
    return {
        "next_question": generate_question(profile),
        "queue": queue,
        "remaining": len(state["missing"]),
        "completeness": state,
        "blocked_councils": sorted({
            council
            for field in state["missing"]
            for council in QUESTION_BANK[field]["unlocks"]
        }) if not state["can_advise"] else [],
    }
