"""
The eight workflows from the deck's orchestration diagram.

Each is a declaration -- which agents, whether to simulate, whether to convene
the councils -- and `base.build_workflow` turns it into a graph. Keeping them as
data means the router, the API and the tests all read the same source of truth
rather than three drifting copies.

Two judgement calls encoded here:

* **Deliberation is expensive**, so only workflows with a genuine trade-off
  convene the councils. "Which credit card?" has a computable answer and does
  not need five councils arguing; "we're getting married" does.
* **Simulation is for decisions with a future**, so it runs where the user is
  choosing between paths, not where they are asking a question of fact.
"""

from __future__ import annotations

from typing import Any

from .base import AGENT_ORDER, build_workflow

#: name -> declaration. `agents` is dependency-sorted by the builder.
WORKFLOWS: dict[str, dict[str, Any]] = {
    "credit_card": {
        "label": "Credit Card Advisory",
        "intent": "credit_card",
        "agents": ("credit_card",),
        "simulate": False,
        "deliberate": False,
        "description": (
            "User asks a card question -> spend profile -> eligibility -> reward "
            "engine -> ranked recommendation -> memory."
        ),
    },
    "salary_day": {
        "label": "On Salary Day Council",
        "intent": "salary_event",
        "agents": (
            "emergency_fund", "debt_trap", "income_projection", "stability",
            "expense_optimizer", "goal_allocation", "asset_allocation", "retirement",
        ),
        "simulate": True,
        "deliberate": True,
        "description": (
            "Salary credited -> is there a surplus? -> surplus: build wealth and "
            "accelerate goals; deficit: find the root cause and generate a "
            "recovery plan -> memory."
        ),
    },
    "income_simulation": {
        "label": "Future Income Simulation",
        "intent": "future_income",
        "agents": ("income_projection", "stability"),
        "simulate": True,
        "deliberate": False,
        "description": (
            "Holt-Winters and SARIMAX forecast -> Monte Carlo expense bands -> "
            "balance projection -> risk scoring -> memory."
        ),
    },
    "goal_planning": {
        "label": "Goal Planning",
        "intent": "goal_planning",
        "agents": (
            "emergency_fund", "goal_allocation", "retirement", "asset_allocation",
        ),
        "simulate": True,
        "deliberate": True,
        "description": (
            "Goals versus surplus -> reserve runway first -> priority-weighted "
            "allocation -> extend deadlines or reduce targets -> memory."
        ),
    },
    "life_event": {
        "label": "Life Event",
        "intent": "life_event",
        "agents": (
            "emergency_fund", "insurance", "debt_trap", "stability",
            "goal_allocation", "scheme_matching",
        ),
        "simulate": True,
        "deliberate": True,
        "description": (
            "Marriage, a child, job loss or a move -> re-check protection and "
            "runway -> re-plan goals -> claim any entitlements -> memory."
        ),
    },
    "fraud": {
        "label": "Fraud Protection",
        "intent": "fraud_check",
        "agents": ("fraud",),
        "simulate": False,
        "deliberate": False,
        "optimise": False,       # nothing to allocate; this is a verdict, not a plan
        "description": (
            "WHOIS and domain age -> RBI and SEBI checks -> complaints and news -> "
            "scam and MLM phrase detection -> deterministic risk score -> memory."
        ),
    },
    "benefits": {
        "label": "Benefits / Scheme Council",
        "intent": "scheme_check",
        "agents": ("eligibility", "scheme_matching"),
        "simulate": False,
        "deliberate": False,
        "optimise": False,
        "description": (
            "Evaluate every scheme's rules against the profile -> rank by benefit, "
            "need and effort -> name the missing field that unlocks the most -> memory."
        ),
    },
    "financial_resilience": {
        "label": "Financial Resilience",
        "intent": "safety_review",
        "agents": (
            "emergency_fund", "insurance", "debt_trap", "stability",
            "bias_detection", "habit_formation", "nudge_strategy", "literacy",
        ),
        "simulate": True,
        "deliberate": True,
        "description": (
            "Runway, cover and leverage -> behavioural patterns that undermine "
            "them -> nudges and habits that hold -> memory."
        ),
    },
}

#: Intent -> workflow name, so the router's output selects a workflow directly.
INTENT_TO_WORKFLOW: dict[str, str] = {
    spec["intent"]: name for name, spec in WORKFLOWS.items()
}
#: The catch-all: a full review runs everything and convenes the councils.
INTENT_TO_WORKFLOW["financial_planning"] = "full_review"
INTENT_TO_WORKFLOW["general"] = "income_simulation"

# Deliberately every agent in AGENT_ORDER rather than the union of the eight
# workflows above -- that union silently omitted `loan_advisor`, which no single
# workflow happens to route to.
WORKFLOWS["full_review"] = {
    "label": "Full Financial Review",
    "intent": "financial_planning",
    "agents": AGENT_ORDER,
    "simulate": True,
    "deliberate": True,
    "description": "Every council, simulated, deliberated and remembered.",
}

_cache: dict[str, Any] = {}


def get_workflow(name: str):
    """Build (and cache) a workflow graph by name."""
    if name not in WORKFLOWS:
        raise ValueError(f"unknown workflow {name!r}; expected one of {sorted(WORKFLOWS)}")
    if name not in _cache:
        spec = WORKFLOWS[name]
        _cache[name] = build_workflow(
            name,
            spec["agents"],
            simulate=spec.get("simulate", False),
            deliberate=spec.get("deliberate", False),
            optimise=spec.get("optimise", True),
        )
    return _cache[name]


def workflow_for_intent(intent: str) -> str:
    """The workflow a routed intent should run."""
    return INTENT_TO_WORKFLOW.get(intent, "income_simulation")
