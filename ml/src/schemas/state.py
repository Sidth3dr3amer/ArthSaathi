"""
The unified LangGraph run state.

This resolves the schema conflict flagged in the audit: `MultiAgent_Testing.ipynb`
and `EmergencyFundAdvisor_FINAL.ipynb` each defined an incompatible `FinancialState`
with zero overlap, which made composing the deliberation council with any domain
agent impossible.

One state now carries all three concerns:

  1. WHO      -> `profile` (UserProfile)
  2. WHAT     -> `query` / `intent` (what the user asked, where the router sent it)
  3. FINDINGS -> `<agent>_result` keys, deliberation, and the final decision

Agent result keys keep the `<agent>_result` convention already used by
`emergency_fund_node`, so migrated nodes need no change to their return shape.

`total=False` throughout: LangGraph merges partial dicts returned by nodes, so no
node is required to populate the whole state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from .profile import UserProfile

Intent = Literal[
    "credit_card",
    "salary_event",
    "future_income",
    "goal_planning",
    "life_event",
    "fraud_check",
    "scheme_check",
    "safety_review",
    "financial_planning",
    "general",
]

CouncilName = Literal["risk", "growth", "benefits", "behavioral", "cashflow"]


class AgentVerdict(TypedDict, total=False):
    """One agent's contribution to a deliberation."""

    agent: str
    council: str
    stance: str
    rationale: str
    confidence: float
    tokens: int


class FinancialState(TypedDict, total=False):
    # ---------------- Who ----------------
    profile: UserProfile
    user_id: str

    #: Raw transaction history, loaded by a workflow when the Behavioral Council
    #: is routed to. Kept out of UserProfile because it is large and episodic
    #: rather than a durable fact about the person.
    transactions: list[dict[str, Any]]

    # ---------------- What ----------------
    query: str
    intent: Intent
    routed_councils: list[CouncilName]
    routing: dict[str, Any]

    # ---------------- Risk council ----------------
    emergency_fund_result: dict[str, Any]
    insurance_result: dict[str, Any]
    debt_trap_result: dict[str, Any]
    fraud_result: dict[str, Any]

    # ---------------- Growth council ----------------
    asset_allocation_result: dict[str, Any]
    credit_card_result: dict[str, Any]
    loan_advisor_result: dict[str, Any]
    retirement_result: dict[str, Any]

    # ---------------- Benefits council ----------------
    scheme_matching_result: dict[str, Any]
    eligibility_result: dict[str, Any]

    # ---------------- Behavioral council ----------------
    bias_detection_result: dict[str, Any]
    habit_formation_result: dict[str, Any]
    nudge_strategy_result: dict[str, Any]
    literacy_result: dict[str, Any]

    # ---------------- Cashflow council ----------------
    stability_result: dict[str, Any]
    income_projection_result: dict[str, Any]
    expense_optimizer_result: dict[str, Any]
    goal_allocation_result: dict[str, Any]

    # ---------------- Decision layer ----------------
    simulation_result: dict[str, Any]
    counterfactual_result: dict[str, Any]
    utility_result: dict[str, Any]
    final_decision: str
    explanation: str

    # Advisors fan out in parallel, so these accumulate rather than overwrite.
    # Without the `operator.add` reducers LangGraph raises InvalidUpdateError on
    # concurrent writes to the same key.
    verdicts: Annotated[list[AgentVerdict], operator.add]
    critiques: Annotated[list[AgentVerdict], operator.add]

    # ---------------- Memory ----------------
    recalled_memories: list[dict[str, Any]]
    memory_written: bool

    # ---------------- Bookkeeping ----------------
    total_tokens: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]


# Every `<agent>_result` key the state understands. Used by the orchestrator to
# collect whatever the routed councils produced, and asserted in tests so a new
# agent cannot silently write to a key nothing reads.
RESULT_KEYS: tuple[str, ...] = (
    "emergency_fund_result",
    "insurance_result",
    "debt_trap_result",
    "fraud_result",
    "asset_allocation_result",
    "credit_card_result",
    "loan_advisor_result",
    "retirement_result",
    "scheme_matching_result",
    "eligibility_result",
    "bias_detection_result",
    "habit_formation_result",
    "nudge_strategy_result",
    "literacy_result",
    "stability_result",
    "income_projection_result",
    "expense_optimizer_result",
    "goal_allocation_result",
)

COUNCIL_AGENTS: dict[str, tuple[str, ...]] = {
    "risk": ("emergency_fund", "insurance", "debt_trap", "fraud"),
    "growth": ("asset_allocation", "credit_card", "loan_advisor", "retirement"),
    "benefits": ("scheme_matching", "eligibility"),
    "behavioral": ("bias_detection", "habit_formation", "nudge_strategy", "literacy"),
    "cashflow": ("stability", "income_projection", "expense_optimizer", "goal_allocation"),
}


def new_state(profile: UserProfile, query: str = "") -> FinancialState:
    """Build a fresh run state for a user."""
    return FinancialState(
        profile=profile,
        user_id=profile.user_id,
        query=query,
        verdicts=[],
        critiques=[],
        recalled_memories=[],
        errors=[],
        total_tokens=0,
        memory_written=False,
    )
