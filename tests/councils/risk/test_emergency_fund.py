"""
Risk Council -> Emergency Fund Agent.

The golden-value test below pins the exact output captured from a live run of
`EmergencyFundAdvisor_FINAL.ipynb` *before* migration. If the extraction changed
the arithmetic even slightly, that test fails.
"""

from __future__ import annotations

import pytest

from ml.src.councils.risk.emergency_fund import (
    emergency_fund_advisor,
    emergency_fund_node,
)
from ml.src.schemas.state import new_state


# --------------------------------------------------------------------------- #
# Golden value — captured from the notebook pre-migration
# --------------------------------------------------------------------------- #

def test_matches_pre_migration_golden_output():
    """Byte-faithful migration check against the notebook's live output."""
    result = emergency_fund_advisor(
        income=90_000,
        expenses=45_000,
        existing_emergency_fund=50_000,
        job_type="salaried",
        dependents=1,
        has_health_insurance=True,
    )

    assert result["target_months"] == 8
    assert result["essential_expenses"] == 31_500.0
    assert result["target_emergency_fund"] == 252_000.0
    assert result["current_emergency_fund"] == 50_000
    assert result["remaining_gap"] == 202_000.0
    assert result["completion_percent"] == 19.84
    assert result["status"] == "Critical"
    assert result["risk_factor"] == 0.3
    assert result["urgency"] == 0.8
    assert result["monthly_surplus"] == 45_000
    assert result["savings_rate"] == 0.5


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_happy_path_returns_full_contract(salaried_profile):
    result = emergency_fund_advisor(
        income=salaried_profile.monthly_income,
        expenses=salaried_profile.essential_expenses,
        existing_emergency_fund=salaried_profile.existing_emergency_fund,
        job_type=salaried_profile.job_type,
        dependents=salaried_profile.dependents,
        has_health_insurance=salaried_profile.has_health_insurance,
    )
    expected_keys = {
        "target_months", "essential_expenses", "target_emergency_fund",
        "current_emergency_fund", "remaining_gap", "completion_percent",
        "status", "risk_factor", "urgency", "monthly_surplus", "savings_rate",
        "priority_score", "emergency_allocation_percent",
        "monthly_emergency_contribution", "monthly_investment_contribution",
        "cash_target", "liquid_fund_target", "months_to_goal",
    }
    assert set(result) == expected_keys


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #

def test_zero_income_and_expenses_does_not_divide_by_zero():
    result = emergency_fund_advisor(
        income=0, expenses=0, existing_emergency_fund=0, job_type="salaried"
    )
    assert result["target_emergency_fund"] == 0
    assert result["completion_percent"] == 100
    assert result["status"] == "Fully Prepared"
    assert result["savings_rate"] == 0
    assert result["months_to_goal"] is None


def test_expenses_above_income_yields_no_surplus():
    result = emergency_fund_advisor(
        income=40_000, expenses=55_000, existing_emergency_fund=0, job_type="freelancer"
    )
    assert result["monthly_surplus"] == 0
    assert result["monthly_emergency_contribution"] == 0
    assert result["months_to_goal"] is None


def test_fully_funded_user_is_capped_at_100_percent():
    result = emergency_fund_advisor(
        income=200_000, expenses=50_000,
        existing_emergency_fund=10_000_000, job_type="govt",
    )
    assert result["completion_percent"] == 100
    assert result["status"] == "Fully Prepared"
    assert result["remaining_gap"] == 0
    assert result["urgency"] == 0


@pytest.mark.parametrize(
    "job_type,dependents,insured,expected_months",
    [
        ("salaried", 0, True, 6),
        ("salaried", 1, True, 8),
        ("salaried", 3, True, 9),
        ("salaried", 0, False, 9),
        ("business", 3, False, 12),
        ("unknown_job", 0, True, 6),   # falls back to the 6-month baseline
    ],
)
def test_target_months_widens_with_risk(job_type, dependents, insured, expected_months):
    result = emergency_fund_advisor(
        income=100_000, expenses=50_000, existing_emergency_fund=0,
        job_type=job_type, dependents=dependents, has_health_insurance=insured,
    )
    assert result["target_months"] == expected_months


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("profile_name", ["salaried_profile", "indebted_profile", "zero_profile"])
def test_invariants_hold_for_every_profile(profile_name, request):
    profile = request.getfixturevalue(profile_name)
    result = emergency_fund_advisor(
        income=profile.monthly_income,
        expenses=profile.essential_expenses,
        existing_emergency_fund=profile.existing_emergency_fund,
        job_type=profile.job_type,
        dependents=profile.dependents,
        has_health_insurance=profile.has_health_insurance,
    )
    assert 0 <= result["completion_percent"] <= 100
    assert 0 <= result["urgency"] <= 1
    assert 0 <= result["risk_factor"] <= 1
    assert 0 <= result["priority_score"] <= 1
    assert 5.0 <= result["emergency_allocation_percent"] <= 70.0
    assert result["remaining_gap"] >= 0
    assert result["monthly_surplus"] >= 0
    assert result["monthly_investment_contribution"] >= 0
    assert result["cash_target"] + result["liquid_fund_target"] == pytest.approx(
        result["target_emergency_fund"], rel=1e-6
    )


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_reads_unified_profile_and_writes_one_key(salaried_profile):
    state = new_state(salaried_profile, query="am I covered for an emergency?")
    patch = emergency_fund_node(state)

    assert set(patch) == {"emergency_fund_result"}
    assert patch["emergency_fund_result"]["status"] == "Critical"


def test_node_output_matches_direct_call(indebted_profile):
    state = new_state(indebted_profile)
    via_node = emergency_fund_node(state)["emergency_fund_result"]
    direct = emergency_fund_advisor(
        income=indebted_profile.monthly_income,
        expenses=indebted_profile.essential_expenses,
        existing_emergency_fund=indebted_profile.existing_emergency_fund,
        job_type=indebted_profile.job_type,
        dependents=indebted_profile.dependents,
        has_health_insurance=indebted_profile.has_health_insurance,
    )
    assert via_node == direct
