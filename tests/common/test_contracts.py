"""
Cross-cutting contracts every agent must honour.

These are the guard-rails that keep 40+ agents composable. They caught a real
bug during Day 1: `detect_recurring` returned `np.bool_`, which silently breaks
`json.dumps` and would have surfaced only at the API layer on Day 6.
"""

from __future__ import annotations

import json

import pytest

from ml.src.councils.behavioral.bias_detection import bias_detection_node
from ml.src.councils.behavioral.habit_formation import habit_formation_node
from ml.src.councils.behavioral.literacy import literacy_node
from ml.src.councils.behavioral.nudge_strategy import nudge_strategy_node
from ml.src.councils.benefits.eligibility import eligibility_node
from ml.src.councils.benefits.scheme_matching import scheme_matching_node
from ml.src.councils.cashflow.expense_optimizer import expense_optimizer_node
from ml.src.councils.cashflow.goal_allocation import goal_allocation_node
from ml.src.councils.cashflow.income_projection import income_projection_node
from ml.src.councils.cashflow.stability import stability_node
from ml.src.councils.growth.asset_allocation import asset_allocation_node
from ml.src.councils.growth.credit_card import credit_card_node
from ml.src.councils.growth.loan_advisor import loan_advisor_node
from ml.src.councils.growth.retirement import retirement_node
from ml.src.councils.risk.debt_trap import debt_trap_node
from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.councils.risk.fraud import fraud_node
from ml.src.councils.risk.insurance import insurance_node
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import COUNCIL_AGENTS, RESULT_KEYS, new_state

# Every council node, paired with the key it is contracted to write.
# Adding an agent here subjects it to every cross-cutting guarantee below.
MIGRATED_NODES = [
    # Risk
    (emergency_fund_node, "emergency_fund_result"),
    (debt_trap_node, "debt_trap_result"),
    (insurance_node, "insurance_result"),
    (fraud_node, "fraud_result"),
    # Growth
    (asset_allocation_node, "asset_allocation_result"),
    (credit_card_node, "credit_card_result"),
    (loan_advisor_node, "loan_advisor_result"),
    (retirement_node, "retirement_result"),
    # Benefits
    (scheme_matching_node, "scheme_matching_result"),
    (eligibility_node, "eligibility_result"),
    # Behavioral
    (bias_detection_node, "bias_detection_result"),
    (habit_formation_node, "habit_formation_result"),
    (nudge_strategy_node, "nudge_strategy_result"),
    (literacy_node, "literacy_result"),
    # Cashflow
    (income_projection_node, "income_projection_result"),
    (stability_node, "stability_result"),
    (expense_optimizer_node, "expense_optimizer_result"),
    (goal_allocation_node, "goal_allocation_result"),
]


@pytest.mark.parametrize("node,key", MIGRATED_NODES, ids=lambda v: getattr(v, "__name__", v))
def test_node_writes_its_declared_result_key(node, key, history_profile):
    patch = node(new_state(history_profile))
    assert key in patch


@pytest.mark.parametrize("node,key", MIGRATED_NODES, ids=lambda v: getattr(v, "__name__", v))
def test_node_output_is_json_serialisable(node, key, history_profile):
    """
    No numpy scalars, no pandas objects, no datetimes may leak out of an agent.
    The memory layer and the API both serialise these payloads verbatim.
    """
    patch = node(new_state(history_profile))
    json.dumps(patch)


@pytest.mark.parametrize("node,key", MIGRATED_NODES, ids=lambda v: getattr(v, "__name__", v))
def test_node_only_writes_keys_the_state_declares(node, key, history_profile):
    known = set(RESULT_KEYS) | {"simulation_result", "errors", "total_tokens"}
    patch = node(new_state(history_profile))
    assert set(patch) <= known, f"{node.__name__} writes undeclared keys: {set(patch) - known}"


@pytest.mark.parametrize("node,key", MIGRATED_NODES, ids=lambda v: getattr(v, "__name__", v))
def test_node_does_not_mutate_the_profile(node, key, history_profile):
    before = history_profile.model_dump()
    node(new_state(history_profile))
    assert history_profile.model_dump() == before


@pytest.mark.parametrize("node,key", MIGRATED_NODES, ids=lambda v: getattr(v, "__name__", v))
def test_node_survives_an_all_zero_profile(node, key, zero_profile):
    """No agent may raise on a brand-new user with no data."""
    patch = node(new_state(zero_profile))
    json.dumps(patch)


# --------------------------------------------------------------------------- #
# State / profile schema
# --------------------------------------------------------------------------- #

def test_every_council_agent_has_a_declared_result_key():
    for council, agents in COUNCIL_AGENTS.items():
        for agent in agents:
            assert f"{agent}_result" in RESULT_KEYS, f"{council}.{agent} has no result key"


def test_result_keys_are_unique():
    assert len(RESULT_KEYS) == len(set(RESULT_KEYS))


def test_new_state_seeds_the_collection_fields():
    state = new_state(UserProfile(user_id="u"), query="hello")
    assert state["query"] == "hello"
    assert state["user_id"] == "u"
    assert state["verdicts"] == []
    assert state["errors"] == []
    assert state["memory_written"] is False


def test_profile_surplus_subtracts_mandatory_debt_servicing():
    profile = UserProfile(
        monthly_income=100_000, essential_expenses=40_000,
        debts=[
            Debt(name="card", debt_type="credit_card",
                 outstanding_amount=50_000, minimum_due=5_000),
            Debt(name="loan", debt_type="personal_loan",
                 outstanding_amount=200_000, emi=8_000),
        ],
    )
    assert profile.monthly_surplus == 100_000 - 40_000 - 13_000
    assert profile.total_debt == 250_000
    assert profile.debt_to_income == pytest.approx(0.13)


def test_profile_handles_zero_income_without_dividing_by_zero():
    assert UserProfile(monthly_income=0).debt_to_income == 0.0


def test_profile_round_trips_through_json():
    profile = UserProfile(
        user_id="rt", monthly_income=50_000,
        debts=[Debt(name="c", debt_type="credit_card", outstanding_amount=1_000)],
    )
    assert UserProfile.model_validate_json(profile.model_dump_json()) == profile
