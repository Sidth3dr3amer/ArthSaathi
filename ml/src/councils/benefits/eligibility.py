"""
Benefits Council -> Eligibility Agent.

New agent. Evaluates a user against the machine-readable eligibility rules in
`ml/data/schemes.json` and returns an auditable trace.

Why a rule engine rather than an LLM: eligibility is a legal determination. A
user told they qualify for PM-KISAN and then rejected at the bank has been
actively harmed. Every verdict here is deterministic, reproducible, and carries
the specific rule that decided it -- which is also what the deck's
"AI Recommendation Basis" panel needs to render.

A rule that cannot be evaluated (the profile lacks the field) is reported as
`unknown` rather than silently passing or failing, so the UI can ask for the
missing detail instead of guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...common import config
from ...schemas.profile import UserProfile
from ...schemas.state import FinancialState

Verdict = str  # "eligible" | "ineligible" | "unknown"


def load_schemes(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the scheme catalogue. Returns [] when the file is absent."""
    path = Path(path or config.SCHEMES_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data.get("schemes", [])


def _check_rule(
    rule: str, expected: Any, profile: UserProfile
) -> tuple[Verdict, str]:
    """Evaluate one rule. Returns (verdict, human-readable reason)."""

    # --- membership rules ------------------------------------------------
    if rule in ("occupation", "job_type", "gender", "social_category"):
        actual = getattr(profile, rule, None)
        if actual is None:
            return "unknown", f"{rule} not recorded"
        allowed = expected if isinstance(expected, list) else [expected]
        if actual in allowed:
            return "eligible", f"{rule} '{actual}' is eligible"
        return "ineligible", f"{rule} '{actual}' not in {allowed}"

    if rule == "residence":
        if profile.residence is None:
            return "unknown", "residence not recorded"
        if profile.residence == expected:
            return "eligible", f"residence '{profile.residence}' matches"
        return "ineligible", f"requires {expected} residence, profile is {profile.residence}"

    # --- numeric bounds ---------------------------------------------------
    if rule == "age_min":
        if profile.age >= expected:
            return "eligible", f"age {profile.age} >= {expected}"
        return "ineligible", f"age {profile.age} is below the minimum of {expected}"

    if rule == "age_max":
        if profile.age <= expected:
            return "eligible", f"age {profile.age} <= {expected}"
        return "ineligible", f"age {profile.age} exceeds the maximum of {expected}"

    if rule == "land_holding_max_ha":
        if profile.land_holding_ha is None:
            return "unknown", "land holding not recorded"
        if profile.land_holding_ha <= expected:
            return "eligible", f"land {profile.land_holding_ha} ha <= {expected} ha"
        return "ineligible", f"land {profile.land_holding_ha} ha exceeds {expected} ha"

    if rule == "annual_income_max":
        income = profile.annual_household_income
        if income is None:
            income = profile.monthly_income * 12 if profile.monthly_income else None
        if income is None:
            return "unknown", "household income not recorded"
        if income <= expected:
            return "eligible", f"income Rs {income:,.0f} <= Rs {expected:,.0f}"
        return "ineligible", f"income Rs {income:,.0f} exceeds Rs {expected:,.0f}"

    if rule == "monthly_income_max":
        if profile.monthly_income <= expected:
            return "eligible", f"monthly income <= Rs {expected:,.0f}"
        return "ineligible", f"monthly income exceeds Rs {expected:,.0f}"

    # --- boolean flag rules -----------------------------------------------
    if rule == "requires":
        for flag in expected:
            if not getattr(profile, flag, False):
                return "ineligible", f"requires {flag.replace('_', ' ')}"
        return "eligible", f"holds {', '.join(expected)}"

    if rule == "excluded_if":
        for flag in expected:
            if getattr(profile, flag, False):
                return "ineligible", f"excluded because {flag.replace('_', ' ')}"
        return "eligible", "no exclusions apply"

    if rule == "any_of":
        reasons = []
        for clause in expected:
            verdicts = [_check_rule(k, v, profile) for k, v in clause.items()]
            if all(v == "eligible" for v, _ in verdicts):
                return "eligible", "; ".join(r for _, r in verdicts)
            reasons.extend(r for _, r in verdicts)
        return "ineligible", f"none of the alternatives matched ({'; '.join(reasons)})"

    # --- unmodelled flags -------------------------------------------------
    # e.g. has_girl_child_under_10, which the profile does not capture yet.
    actual = getattr(profile, rule, None)
    if actual is None:
        return "unknown", f"{rule.replace('_', ' ')} not recorded"
    return ("eligible", f"{rule} satisfied") if actual == expected else (
        "ineligible", f"{rule} is {actual}, needs {expected}"
    )


def check_scheme(scheme: dict[str, Any], profile: UserProfile) -> dict[str, Any]:
    """Evaluate one scheme. Every rule is reported, not just the failing one."""
    rules = scheme.get("eligibility", {}) or {}
    checks: list[dict[str, str]] = []

    for rule, expected in rules.items():
        verdict, reason = _check_rule(rule, expected, profile)
        checks.append({"rule": rule, "verdict": verdict, "reason": reason})

    failed = [c for c in checks if c["verdict"] == "ineligible"]
    unknown = [c for c in checks if c["verdict"] == "unknown"]

    if failed:
        verdict = "ineligible"
    elif unknown:
        verdict = "unknown"
    else:
        verdict = "eligible"

    # Confidence reflects how much of the rule set we could actually evaluate.
    known = len(checks) - len(unknown)
    confidence = round(known / len(checks), 2) if checks else 1.0

    return {
        "scheme_id": scheme.get("scheme_id"),
        "name": scheme.get("name"),
        "category": scheme.get("category"),
        "benefit": scheme.get("benefit", {}),
        "url": scheme.get("url"),
        "documents": scheme.get("documents", []),
        # Propagated so downstream agents (scheme matching) can reason about who
        # a scheme targets, not just whether this user passed.
        "eligibility": rules,
        "verdict": verdict,
        "confidence": confidence,
        "checks": checks,
        "blocking_reasons": [c["reason"] for c in failed],
        "missing_information": [c["rule"] for c in unknown],
    }


def eligibility_advisor(
    profile: UserProfile, schemes: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Evaluate every scheme. Pure and deterministic: no I/O beyond the catalogue."""
    catalogue = schemes if schemes is not None else load_schemes()
    results = [check_scheme(s, profile) for s in catalogue]

    eligible = [r for r in results if r["verdict"] == "eligible"]
    unknown = [r for r in results if r["verdict"] == "unknown"]
    ineligible = [r for r in results if r["verdict"] == "ineligible"]

    # Which single missing profile field would unlock the most schemes.
    missing_counts: dict[str, int] = {}
    for row in unknown:
        for field in row["missing_information"]:
            missing_counts[field] = missing_counts.get(field, 0) + 1
    most_valuable = sorted(missing_counts.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "schemes_evaluated": len(results),
        "eligible": eligible,
        "possibly_eligible": unknown,
        "ineligible": ineligible,
        "eligible_count": len(eligible),
        "possibly_eligible_count": len(unknown),
        "profile_completeness": round(
            1 - (len(missing_counts) / 10), 2
        ) if missing_counts else 1.0,
        "ask_user_for": [field for field, _ in most_valuable[:3]],
        "results": results,
    }


def eligibility_node(
    state: FinancialState, schemes: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """LangGraph adapter."""
    return {"eligibility_result": eligibility_advisor(state["profile"], schemes)}
