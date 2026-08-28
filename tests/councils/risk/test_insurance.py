"""Risk Council -> Insurance Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.risk.insurance import (
    HEALTH_BASE_LAKH,
    LAKH,
    _term_multiple,
    insurance_advisor,
    insurance_node,
)
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import new_state


def advise(**overrides):
    base = dict(monthly_income=90_000, age=35, dependents=2,
                has_health_insurance=False, total_debt=0.0)
    base.update(overrides)
    return insurance_advisor(**base)


# --------------------------------------------------------------------------- #
# Term life
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "age,multiple", [(25, 15), (35, 12), (45, 10), (55, 7), (65, 5)]
)
def test_term_multiple_falls_with_age(age, multiple):
    """Younger earners have more earning years to replace."""
    assert _term_multiple(age) == multiple


def test_term_cover_is_income_multiple_plus_debt():
    out = advise(age=35, dependents=2, total_debt=140_000)
    expected = 90_000 * 12 * 12 + 140_000
    assert out["covers"]["term_life"]["required"] == pytest.approx(expected)


def test_no_dependants_and_no_debt_needs_no_term_cover():
    """Term life replaces income for people who depend on it."""
    out = advise(dependents=0, total_debt=0)
    assert out["covers"]["term_life"]["required"] == 0
    assert out["covers"]["term_life"]["gap"] == 0


def test_debt_alone_justifies_term_cover():
    out = advise(dependents=0, total_debt=500_000)
    assert out["covers"]["term_life"]["required"] > 0


def test_existing_cover_reduces_the_gap():
    out = advise(dependents=2, existing_term_cover=5_000_000)
    cover = out["covers"]["term_life"]
    assert cover["gap"] == pytest.approx(cover["required"] - 5_000_000)


def test_overinsured_user_has_no_negative_gap():
    out = advise(dependents=2, existing_term_cover=999_000_000)
    assert out["covers"]["term_life"]["gap"] == 0


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dependents", [0, 1, 2, 3])
def test_health_floater_scales_with_dependants(dependents):
    out = advise(age=30, dependents=dependents)
    assert out["covers"]["health"]["required"] == HEALTH_BASE_LAKH[dependents] * LAKH


def test_health_requirement_is_loaded_after_45():
    younger = advise(age=40, dependents=1)["covers"]["health"]["required"]
    older = advise(age=50, dependents=1)["covers"]["health"]["required"]
    assert older == pytest.approx(younger * 1.5)


def test_many_dependants_are_capped():
    out = advise(dependents=9)
    assert out["covers"]["health"]["required"] <= 15 * LAKH * 1.5


# --------------------------------------------------------------------------- #
# Critical illness
# --------------------------------------------------------------------------- #

def test_critical_illness_only_applies_from_35():
    assert advise(age=30)["covers"]["critical_illness"]["required"] == 0
    assert advise(age=35)["covers"]["critical_illness"]["required"] == 90_000 * 12


# --------------------------------------------------------------------------- #
# Prioritisation
# --------------------------------------------------------------------------- #

def test_missing_health_cover_outranks_a_larger_term_gap():
    """Uninsured medical risk is weighted above a bigger but insured-against gap."""
    out = advise(age=30, dependents=1, has_health_insurance=False, monthly_income=20_000)
    assert out["priority_cover"] == "health"


def test_held_health_cover_lets_term_take_priority():
    out = advise(age=35, dependents=2, has_health_insurance=True,
                 existing_health_cover=10 * LAKH)
    assert out["priority_cover"] == "term_life"


def test_fully_covered_user_has_no_priority():
    out = insurance_advisor(
        monthly_income=90_000, age=25, dependents=0,
        has_health_insurance=True, existing_health_cover=999_000_000,
    )
    assert out["priority_cover"] is None
    assert out["status"] == "Fully Covered"


# --------------------------------------------------------------------------- #
# Status / invariants
# --------------------------------------------------------------------------- #

def test_exposure_is_bounded_and_status_consistent():
    for age in (25, 35, 45, 55):
        for dependents in (0, 2):
            out = advise(age=age, dependents=dependents)
            assert 0.0 <= out["exposure"] <= 1.0
            assert out["status"] in {
                "Severely Underinsured", "Underinsured", "Partially Covered",
                "Adequately Covered", "Fully Covered",
            }


def test_uninsured_user_is_severely_underinsured():
    assert advise(has_health_insurance=False, dependents=2)["status"] == "Severely Underinsured"


def test_premium_scales_with_age():
    young = advise(age=25, dependents=2)["total_annual_premium_estimate"]
    old = advise(age=55, dependents=2)["total_annual_premium_estimate"]
    assert old > young


def test_zero_income_does_not_divide_by_zero():
    out = insurance_advisor(monthly_income=0, age=30, dependents=1,
                            has_health_insurance=False)
    assert out["premium_as_income_percent"] == 0.0
    assert 0 <= out["exposure"] <= 1


def test_recommendations_cover_every_gap():
    out = advise(dependents=2, has_health_insurance=False)
    assert len(out["recommendations"]) == len(
        [c for c in out["covers"].values() if c["gap"] > 0]
    )


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(indebted_profile):
    patch = insurance_node(new_state(indebted_profile))
    assert set(patch) == {"insurance_result"}
    json.dumps(patch)


def test_node_includes_debt_in_the_term_requirement(indebted_profile):
    result = insurance_node(new_state(indebted_profile))["insurance_result"]
    assert result["covers"]["term_life"]["required"] > indebted_profile.monthly_income * 12


def test_node_survives_the_zero_profile(zero_profile):
    result = insurance_node(new_state(zero_profile))["insurance_result"]
    assert 0 <= result["exposure"] <= 1
