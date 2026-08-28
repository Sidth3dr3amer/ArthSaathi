"""Cashflow Council -> Goal Allocation Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.cashflow.goal_allocation import (
    EMERGENCY_FIRST_SHARE,
    PRIORITY_WEIGHT,
    goal_allocation_advisor,
    goal_allocation_node,
)
from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.schemas.profile import Goal, UserProfile
from ml.src.schemas.state import new_state

HOME = {"name": "Home", "target_amount": 1_500_000, "current_amount": 200_000,
        "target_months": 48, "priority": "high"}
CAR = {"name": "Car", "target_amount": 600_000, "current_amount": 50_000,
       "target_months": 36, "priority": "low"}


# --------------------------------------------------------------------------- #
# Feasible plans
# --------------------------------------------------------------------------- #

def test_ample_surplus_funds_every_goal_in_full():
    out = goal_allocation_advisor([HOME, CAR], monthly_surplus=200_000)
    assert out["feasible"] is True
    assert out["status"] == "All goals fundable"
    assert out["shortfall"] == 0
    for goal in out["goals"]:
        assert goal["allocated_monthly"] == pytest.approx(goal["required_monthly"])
        assert goal["on_track"] is True


def test_required_monthly_is_remaining_over_months():
    out = goal_allocation_advisor([HOME], monthly_surplus=200_000)
    goal = out["goals"][0]
    assert goal["remaining"] == 1_300_000
    assert goal["required_monthly"] == pytest.approx(1_300_000 / 48)


def test_progress_percent_is_reported():
    out = goal_allocation_advisor([HOME], monthly_surplus=200_000)
    assert out["goals"][0]["progress_percent"] == pytest.approx(13.33, abs=0.01)


# --------------------------------------------------------------------------- #
# Infeasible plans
# --------------------------------------------------------------------------- #

def test_insufficient_surplus_is_reported_honestly():
    out = goal_allocation_advisor([HOME, CAR], monthly_surplus=20_000)
    assert out["feasible"] is False
    assert out["status"] == "Goals exceed surplus"
    assert out["shortfall"] > 0


def test_allocation_is_priority_weighted_when_short():
    out = goal_allocation_advisor([HOME, CAR], monthly_surplus=20_000)
    by_name = {g["name"]: g for g in out["goals"]}
    ratio = by_name["Home"]["allocated_monthly"] / by_name["Car"]["allocated_monthly"]
    assert ratio == pytest.approx(PRIORITY_WEIGHT["high"] / PRIORITY_WEIGHT["low"], rel=1e-6)


def test_allocation_never_exceeds_the_allocatable_surplus():
    out = goal_allocation_advisor([HOME, CAR], monthly_surplus=20_000)
    assert sum(g["allocated_monthly"] for g in out["goals"]) <= out["allocatable"] + 0.01


def test_both_escape_routes_are_offered_when_infeasible():
    """Extend the deadline, or lower the target -- the user chooses."""
    out = goal_allocation_advisor([HOME, CAR], monthly_surplus=20_000)
    assert len(out["required_extension"]) == 2
    assert len(out["feasible_targets"]) == 2
    for row in out["required_extension"]:
        assert row["required_months"] > row["original_months"]
    for row in out["feasible_targets"]:
        assert row["feasible_target"] < row["original_target"]


def test_no_surplus_at_all_is_its_own_status():
    out = goal_allocation_advisor([HOME], monthly_surplus=0)
    assert out["status"] == "No surplus to allocate"
    assert out["goals"][0]["allocated_monthly"] == 0
    assert out["goals"][0]["months_to_goal"] is None


def test_negative_surplus_is_treated_as_zero():
    out = goal_allocation_advisor([HOME], monthly_surplus=-5_000)
    assert out["monthly_surplus"] == 0


# --------------------------------------------------------------------------- #
# Emergency fund has first claim
# --------------------------------------------------------------------------- #

def test_emergency_gap_reserves_part_of_the_surplus():
    out = goal_allocation_advisor(
        [HOME], monthly_surplus=40_000,
        emergency_gap=200_000, emergency_monthly_contribution=15_000,
    )
    assert out["reserved_for_emergency"] == 15_000
    assert out["allocatable"] == 25_000


def test_reservation_is_capped_at_half_the_surplus():
    """A huge suggested contribution cannot starve every goal."""
    out = goal_allocation_advisor(
        [HOME], monthly_surplus=40_000,
        emergency_gap=999_999, emergency_monthly_contribution=999_999,
    )
    assert out["reserved_for_emergency"] == 40_000 * EMERGENCY_FIRST_SHARE


def test_a_funded_emergency_fund_reserves_nothing():
    out = goal_allocation_advisor([HOME], monthly_surplus=40_000, emergency_gap=0)
    assert out["reserved_for_emergency"] == 0
    assert out["allocatable"] == 40_000


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_no_goals_is_a_valid_state():
    out = goal_allocation_advisor([], monthly_surplus=50_000)
    assert out["status"] == "No goals set"
    assert out["goals"] == []
    assert out["feasible"] is True


def test_an_already_met_goal_needs_no_allocation():
    met = {"name": "Done", "target_amount": 100_000, "current_amount": 100_000,
           "target_months": 12, "priority": "high"}
    out = goal_allocation_advisor([met], monthly_surplus=50_000)
    goal = out["goals"][0]
    assert goal["remaining"] == 0
    assert goal["allocated_monthly"] == 0
    assert goal["on_track"] is True


def test_goal_with_no_deadline_does_not_divide_by_zero():
    out = goal_allocation_advisor(
        [{"name": "Someday", "target_amount": 100_000, "current_amount": 0,
          "target_months": 0, "priority": "medium"}],
        monthly_surplus=50_000,
    )
    assert out["goals"][0]["required_monthly"] == 100_000


def test_unknown_priority_falls_back_to_medium_weight():
    out = goal_allocation_advisor(
        [{**HOME, "priority": "urgent-ish"}], monthly_surplus=10_000
    )
    assert out["goals"][0]["weight"] == 2.0


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(indebted_profile):
    patch = goal_allocation_node(new_state(indebted_profile))
    assert set(patch) == {"goal_allocation_result"}
    json.dumps(patch)


def test_node_consumes_the_emergency_fund_agent_upstream():
    """The two agents must compose: runway is reserved before goals are funded."""
    profile = UserProfile(
        user_id="compose", monthly_income=90_000, essential_expenses=45_000,
        existing_emergency_fund=30_000, dependents=2,
        goals=[Goal(name="Home", target_amount=1_500_000,
                    current_amount=200_000, target_months=48, priority="high")],
    )
    state = new_state(profile)
    state.update(emergency_fund_node(state))
    result = goal_allocation_node(state)["goal_allocation_result"]
    assert result["reserved_for_emergency"] > 0


def test_node_without_an_upstream_emergency_result_reserves_nothing(salaried_profile):
    result = goal_allocation_node(new_state(salaried_profile))["goal_allocation_result"]
    assert result["reserved_for_emergency"] == 0


def test_node_survives_the_zero_profile(zero_profile):
    result = goal_allocation_node(new_state(zero_profile))["goal_allocation_result"]
    assert result["status"] == "No goals set"
