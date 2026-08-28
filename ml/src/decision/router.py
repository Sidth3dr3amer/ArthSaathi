"""
Decision Layer -> Intent Router / Decision Orchestrator.

The box at the top of the deck's orchestration diagram. Everything downstream
depends on it, and until now it did not exist -- which is why a routine
"give me a financial review" fired eight live fraud searches.

Routing is deterministic first, LLM second:

  1. Pattern rules over the query. Fast, free, reproducible, testable. Most
     real queries are unambiguous and never need a model.
  2. An LLM classifier, only when the rules are inconclusive, and only when a
     provider is configured. Its answer is validated against the known intent
     set -- a hallucinated intent falls back to `general` rather than crashing
     a workflow.

The router returns the councils to activate AND the specific agents, because
activating a whole council to answer "is this a scam" wastes latency and, for
the network-bound agents, money.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from ..common import llm
from ..schemas.state import COUNCIL_AGENTS, FinancialState, Intent

#: intent -> (councils, specific agents). Agents are a subset of those councils
#: plus decision-layer helpers.
INTENT_ROUTES: dict[str, dict[str, tuple[str, ...]]] = {
    "credit_card": {
        "councils": ("growth",),
        "agents": ("credit_card",),
    },
    "salary_event": {
        "councils": ("cashflow", "risk", "growth"),
        "agents": ("stability", "income_projection", "emergency_fund",
                   "debt_trap", "goal_allocation", "asset_allocation"),
    },
    "future_income": {
        "councils": ("cashflow",),
        "agents": ("income_projection", "stability"),
    },
    "goal_planning": {
        "councils": ("cashflow", "growth"),
        "agents": ("goal_allocation", "emergency_fund", "retirement", "asset_allocation"),
    },
    "life_event": {
        "councils": ("risk", "cashflow", "growth"),
        "agents": ("emergency_fund", "insurance", "stability", "goal_allocation"),
    },
    "fraud_check": {
        "councils": ("risk",),
        "agents": ("fraud",),
    },
    "scheme_check": {
        "councils": ("benefits",),
        "agents": ("eligibility", "scheme_matching"),
    },
    "safety_review": {
        "councils": ("risk",),
        "agents": ("emergency_fund", "insurance", "debt_trap"),
    },
    "financial_planning": {
        "councils": ("risk", "growth", "benefits", "behavioral", "cashflow"),
        "agents": (),          # empty means "every agent in the routed councils"
    },
    "general": {
        "councils": ("cashflow",),
        "agents": ("stability",),
    },
}

#: Ordered rules -- the first match wins, so put specific intents above broad ones.
INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fraud_check", re.compile(
        r"\b(scam|fraud|fraudulent|ponzi|chit fund|mlm|legit|legitimate|genuine|"
        r"too good to be true|guaranteed returns?|double (my|your) money|"
        r"is this (safe|real)|should i trust|verify this (company|scheme|offer))\b", re.I)),
    ("credit_card", re.compile(
        r"\b(credit card|which card|best card|cashback card|reward points?|"
        r"lounge access|annual fee|card recommendation)\b", re.I)),
    ("scheme_check", re.compile(
        r"\b(scheme|subsidy|subsidies|government|govt|yojana|pm-?kisan|pmjay|"
        r"ayushman|entitle(d|ment)|benefits? am i|eligible for)\b", re.I)),
    # `.{0,15}` tolerates the filler people actually type -- "salary just got
    # credited", "salary has finally come in" -- which strict adjacency missed.
    ("salary_event", re.compile(
        r"\b(salary.{0,15}\b(credited|came|come|landed|arrived|in)\b|"
        r"got paid|pay ?day|received my (salary|pay)|"
        r"this month'?s (salary|pay)|monthly plan|month(ly)? budget)\b", re.I)),
    ("life_event", re.compile(
        r"\b(getting married|marriage|wedding|baby|child birth|pregnan|"
        r"lost my job|laid off|job loss|moving (house|city)|relocat|"
        r"buying a (house|home|car)|medical emergency|surgery)\b", re.I)),
    # `save .{0,20} for` tolerates an amount in between -- "save 20 lakh for a
    # house" is the way people actually phrase a goal.
    ("goal_planning", re.compile(
        r"\b(goal|sav(e|ing).{0,20}?\bfor\b|down payment|target amount|"
        r"retire(ment)?|corpus|by when can i|afford to buy)\b", re.I)),
    ("future_income", re.compile(
        r"\b(forecast|project(ion|ed)?|next (few )?months?|future income|"
        r"will i (have|be able)|cash ?flow|runway|how much will i)\b", re.I)),
    ("safety_review", re.compile(
        r"\b(emergency fund|insurance|insured|cover(age)?|protect(ion)?|"
        r"safety net|rainy day|debt trap|too much debt)\b", re.I)),
    ("financial_planning", re.compile(
        r"\b(full (financial )?review|complete (review|analysis|picture)|"
        r"overall|everything|health check|financial health|where do i stand|"
        r"analyse my finances|analyze my finances|financial plan)\b", re.I)),
)

CLASSIFIER_SYSTEM = (
    "You classify a personal-finance question into exactly one intent. "
    "Reply with only the intent name, nothing else. Valid intents: "
    + ", ".join(INTENT_ROUTES)
)


def classify_by_rules(query: str) -> tuple[str | None, str]:
    """Deterministic classification. Returns (intent, matched pattern text)."""
    if not query or not query.strip():
        return None, "empty query"
    for intent, pattern in INTENT_PATTERNS:
        match = pattern.search(query)
        if match:
            return intent, match.group(0)
    return None, "no rule matched"


def classify_by_llm(query: str, provider: llm.Provider = "groq") -> tuple[str | None, str]:
    """
    LLM fallback. Never raises -- a failure or a hallucinated label degrades to
    `None`, and the caller falls back to `general`.
    """
    try:
        raw = llm.chat(query, provider=provider, system=CLASSIFIER_SYSTEM)
    except Exception as exc:
        return None, f"classifier unavailable: {exc!r}"

    candidate = (raw or "").strip().lower().replace(" ", "_").strip(".\"'")
    if candidate in INTENT_ROUTES:
        return candidate, f"llm classified as {candidate}"
    return None, f"llm returned an unknown intent {candidate!r}"


def agents_for(intent: str) -> tuple[str, ...]:
    """Agents to activate. An empty agent list means every agent in the councils."""
    route = INTENT_ROUTES.get(intent, INTENT_ROUTES["general"])
    if route["agents"]:
        return route["agents"]
    agents: list[str] = []
    for council in route["councils"]:
        agents.extend(COUNCIL_AGENTS.get(council, ()))
    return tuple(agents)


def route(
    query: str,
    use_llm: bool = True,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Decide what this query should activate.

    Pure when `use_llm=False`, which is how the tests exercise every rule.
    """
    intent, reason = classify_by_rules(query)
    method = "rules"

    if intent is None and use_llm and query and query.strip():
        intent, reason = classify_by_llm(query, provider)
        method = "llm"

    if intent is None:
        intent, method = "general", method if method == "llm" else "fallback"

    route_spec = INTENT_ROUTES[intent]
    return {
        "intent": intent,
        "councils": list(route_spec["councils"]),
        "agents": list(agents_for(intent)),
        "method": method,
        "reason": reason,
        "requires_simulation": intent in (
            "salary_event", "future_income", "goal_planning", "life_event",
            "financial_planning",
        ),
        "requires_deliberation": intent in (
            "life_event", "goal_planning", "financial_planning",
        ),
    }


def router_node(
    state: FinancialState,
    use_llm: bool = True,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """LangGraph adapter. Writes the routing decision into the state."""
    decision = route(state.get("query", ""), use_llm=use_llm, provider=provider)
    return {
        "intent": decision["intent"],
        "routed_councils": decision["councils"],
        "routing": decision,
    }
