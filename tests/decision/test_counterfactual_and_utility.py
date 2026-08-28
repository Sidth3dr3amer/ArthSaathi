"""Decision Layer -> Counterfactual Simulator and Utility Optimizer."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.cashflow.goal_allocation import goal_allocation_node
from ml.src.councils.growth.retirement import retirement_node
from ml.src.councils.risk.debt_trap import debt_trap_node
from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.councils.risk.insurance import insurance_node
from ml.src.decision.counterfactual import (
    counterfactual_advisor,
    counterfactual_node,
    default_scenarios,
)
from ml.src.decision.utility import (
    STEP,
    build_claims,
    utility_advisor,
    utility_node,
)
from ml.src.schemas.profile import Debt, Goal, UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(
        user_id="dec", age=33, job_type="salaried", monthly_income=95_000,
        essential_expenses=48_000, existing_emergency_fund=40_000,
        retirement_corpus=300_000, dependents=2, has_health_insurance=False,
        debts=[Debt(name="Card", debt_type="credit_card", outstanding_amount=160_000,
                    interest_rate=42.0, minimum_due=8_000)],
        goals=[Goal(name="Home", target_amount=2_000_000, current_amount=250_000,
                    target_months=60, priority="high")],
    )


@pytest.fixture
def loaded_state(profile):
    state = new_state(profile, query="what should I do with my money?")
    for node in (emergency_fund_node, insurance_node, debt_trap_node,
                 goal_allocation_node, retirement_node):
        state.update(node(state))
    return state


# =========================================================================== #
# Counterfactual Simulator
# =========================================================================== #

def test_cutting_spending_improves_every_tracked_metric(profile):
    out = counterfactual_advisor(profile, {"Spend less": {"essential_expenses": "-5000"}})
    result = out["results"][0]
    assert result["net_effect"] == "strictly better"
    assert result["worsened"] == 0


def test_spending_more_makes_things_worse(profile):
    out = counterfactual_advisor(profile, {"Spend more": {"essential_expenses": "+15000"}})
    assert out["results"][0]["net_effect"] == "strictly worse"


def test_relative_deltas_are_applied(profile):
    out = counterfactual_advisor(profile, {"Raise": {"monthly_income": "+10000"}})
    metrics = {m["field"]: m for m in out["results"][0]["metrics"]}
    assert metrics["allocatable"]["after"] > metrics["allocatable"]["before"]


def test_absolute_values_are_applied(profile):
    out = counterfactual_advisor(profile, {"Debt free": {"debts": []}})
    assert out["results"][0]["net_effect"] in ("strictly better", "mixed")


def test_an_unchanged_scenario_reports_no_change(profile):
    out = counterfactual_advisor(profile, {"Nothing": {}})
    assert out["results"][0]["net_effect"] == "no material change"


def test_metrics_carry_direction_and_delta(profile):
    out = counterfactual_advisor(profile, {"Spend less": {"essential_expenses": "-5000"}})
    for metric in out["results"][0]["metrics"]:
        assert metric["direction"] in ("better", "worse", "unchanged")
        assert metric["after"] - metric["before"] == pytest.approx(metric["delta"], abs=0.01)


def test_direction_respects_whether_higher_is_better(profile):
    """A falling gap is better; falling readiness is worse."""
    out = counterfactual_advisor(profile, {"Spend less": {"essential_expenses": "-5000"}})
    by_field = {m["field"]: m for m in out["results"][0]["metrics"]}
    assert by_field["remaining_gap"]["delta"] < 0
    assert by_field["remaining_gap"]["direction"] == "better"


def test_a_broken_scenario_is_isolated(profile):
    out = counterfactual_advisor(profile, {"Bad": {"age": "not-a-number"}})
    assert out["results"][0]["net_effect"] == "error"
    assert "error" in out["results"][0]


def test_best_scenario_is_the_most_improving(profile):
    out = counterfactual_advisor(profile, {
        "Small cut": {"essential_expenses": "-2000"},
        "No change": {},
    })
    assert out["best_scenario"] == "Small cut"


def test_default_scenarios_scale_to_the_user(profile):
    scenarios = default_scenarios(profile)
    assert len(scenarios) == 4
    assert any("Clear all debt" in s for s in scenarios)


def test_counterfactual_node_contract(loaded_state):
    patch = counterfactual_node(loaded_state)
    assert set(patch) == {"counterfactual_result"}
    json.dumps(patch)


def test_counterfactual_does_not_mutate_the_profile(profile):
    before = profile.model_dump()
    counterfactual_advisor(profile, default_scenarios(profile))
    assert profile.model_dump() == before


# =========================================================================== #
# Utility Optimizer
# =========================================================================== #

def test_claims_are_built_from_whatever_ran(loaded_state):
    claims = build_claims(loaded_state)
    kinds = {c["claim"] for c in claims}
    assert {"emergency_fund", "debt_repayment", "insurance"} <= kinds


def test_a_bare_state_produces_no_claims(salaried_profile):
    assert build_claims(new_state(salaried_profile)) == []


def test_prerequisites_are_funded_before_optimisation():
    claims = [{"claim": "goals", "label": "Goals", "base_utility": 0.6,
               "saturation": 50_000, "kind": "aspirational", "rationale": ""}]
    out = utility_advisor(claims, surplus=20_000, mandatory_debt_service=8_000)
    prereq = out["prerequisites"][0]
    assert prereq["fully_met"] is True
    assert prereq["funded"] == 8_000


def test_prerequisites_are_capped_by_available_surplus():
    out = utility_advisor([], surplus=5_000, mandatory_debt_service=8_000)
    assert out["prerequisites"][0]["funded"] == 5_000
    assert out["prerequisites"][0]["fully_met"] is False


def test_allocation_never_exceeds_the_surplus(loaded_state):
    out = utility_node(loaded_state)["utility_result"]
    assert out["total_allocated"] <= out["surplus"] + 0.01
    assert out["unallocated"] >= 0


def test_high_rate_debt_outranks_aspirational_goals(loaded_state):
    """A guaranteed 42% return must beat a home-savings goal."""
    out = utility_node(loaded_state)["utility_result"]
    by_claim = {p["claim"]: p["monthly_allocation"] for p in out["allocation_plan"]}
    assert by_claim.get("debt_repayment", 0) > by_claim.get("goals", 0)


def test_diminishing_returns_produce_a_split_not_winner_takes_all(loaded_state):
    out = utility_node(loaded_state)["utility_result"]
    assert len(out["allocation_plan"]) >= 2
    assert max(p["share_of_surplus"] for p in out["allocation_plan"]) < 0.95


def test_shares_sum_to_at_most_one(loaded_state):
    out = utility_node(loaded_state)["utility_result"]
    assert sum(p["share_of_surplus"] for p in out["allocation_plan"]) <= 1.0001


def test_no_surplus_allocates_nothing():
    out = utility_advisor(
        [{"claim": "x", "label": "X", "base_utility": 1.0, "saturation": 1_000,
          "kind": "protective", "rationale": ""}],
        surplus=0,
    )
    assert out["status"] == "No surplus to allocate"
    assert out["allocation_plan"] == []


def test_no_claims_leaves_the_surplus_unallocated():
    out = utility_advisor([], surplus=30_000)
    assert out["status"] == "No competing claims"
    assert out["unallocated"] == 30_000


def test_allocation_is_granular_to_the_step():
    out = utility_advisor(
        [{"claim": "a", "label": "A", "base_utility": 1.0, "saturation": 100_000,
          "kind": "protective", "rationale": ""}],
        surplus=10_000,
    )
    assert out["allocation_plan"][0]["monthly_allocation"] % STEP == 0


def test_method_is_documented(loaded_state):
    assert "marginal utility" in utility_node(loaded_state)["utility_result"]["method"]


def test_utility_node_contract(loaded_state):
    patch = utility_node(loaded_state)
    assert set(patch) == {"utility_result"}
    json.dumps(patch)


def test_utility_node_survives_the_zero_profile(zero_profile):
    result = utility_node(new_state(zero_profile))["utility_result"]
    assert result["allocation_plan"] == []
