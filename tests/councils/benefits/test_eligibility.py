"""
Benefits Council -> Eligibility Agent.

Eligibility is a legal determination: telling someone they qualify when they do
not causes real harm. These tests pin the rule engine's behaviour, especially
the three-way verdict (eligible / ineligible / unknown) that keeps it from
guessing when the profile is incomplete.
"""

from __future__ import annotations

import json

import pytest

from ml.src.councils.benefits.eligibility import (
    check_scheme,
    eligibility_advisor,
    eligibility_node,
    load_schemes,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture
def farmer() -> UserProfile:
    """The deck's slide-9 persona."""
    return UserProfile(
        user_id="rahul", name="Rahul Patil", age=42, job_type="business",
        occupation="farmer", state="Maharashtra", residence="rural",
        gender="male", social_category="obc",
        monthly_income=35_000, essential_expenses=22_000,
        annual_household_income=420_000, land_holding_ha=1.5, dependents=3,
        has_health_insurance=False,
    )


@pytest.fixture
def catalogue():
    return load_schemes()


PMKISAN = {
    "scheme_id": "TEST-KISAN", "name": "Test Kisan", "category": "income_support",
    "benefit": {"type": "cash_transfer", "amount": 6000, "frequency": "annual"},
    "documents": ["Aadhaar"],
    "eligibility": {
        "occupation": ["farmer"], "land_holding_max_ha": 2, "age_min": 18,
        "requires": ["has_bank_account"], "excluded_if": ["is_income_tax_payer"],
    },
}


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #

def test_catalogue_loads_the_seeded_schemes(catalogue):
    assert len(catalogue) >= 20
    assert all("scheme_id" in s and "eligibility" in s for s in catalogue)


def test_scheme_ids_are_unique(catalogue):
    ids = [s["scheme_id"] for s in catalogue]
    assert len(ids) == len(set(ids))


def test_missing_catalogue_file_returns_empty(tmp_path):
    assert load_schemes(tmp_path / "absent.json") == []


def test_malformed_catalogue_returns_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_schemes(bad) == []


# --------------------------------------------------------------------------- #
# Rule evaluation
# --------------------------------------------------------------------------- #

def test_fully_qualifying_user_is_eligible_with_full_confidence(farmer):
    out = check_scheme(PMKISAN, farmer)
    assert out["verdict"] == "eligible"
    assert out["confidence"] == 1.0
    assert out["blocking_reasons"] == []
    assert all(c["verdict"] == "eligible" for c in out["checks"])


def test_land_ceiling_blocks_a_large_farmer(farmer):
    big = farmer.model_copy(update={"land_holding_ha": 5.0})
    out = check_scheme(PMKISAN, big)
    assert out["verdict"] == "ineligible"
    assert any("exceeds" in r for r in out["blocking_reasons"])


def test_exclusion_flag_blocks(farmer):
    taxpayer = farmer.model_copy(update={"is_income_tax_payer": True})
    out = check_scheme(PMKISAN, taxpayer)
    assert out["verdict"] == "ineligible"
    assert any("income tax payer" in r for r in out["blocking_reasons"])


def test_missing_requirement_blocks(farmer):
    unbanked = farmer.model_copy(update={"has_bank_account": False})
    out = check_scheme(PMKISAN, unbanked)
    assert out["verdict"] == "ineligible"


def test_wrong_occupation_blocks(farmer):
    clerk = farmer.model_copy(update={"occupation": "clerk"})
    assert check_scheme(PMKISAN, clerk)["verdict"] == "ineligible"


def test_unrecorded_field_yields_unknown_not_a_guess(farmer):
    """The engine must never assume; it asks instead."""
    no_land = farmer.model_copy(update={"land_holding_ha": None})
    out = check_scheme(PMKISAN, no_land)
    assert out["verdict"] == "unknown"
    assert "land_holding_max_ha" in out["missing_information"]
    assert out["confidence"] < 1.0


def test_a_hard_failure_outranks_an_unknown(farmer):
    """One definite disqualification settles it even with other fields missing."""
    mixed = farmer.model_copy(update={"land_holding_ha": None, "is_income_tax_payer": True})
    assert check_scheme(PMKISAN, mixed)["verdict"] == "ineligible"


def test_every_rule_is_reported_not_only_the_failing_one(farmer):
    out = check_scheme(PMKISAN, farmer)
    assert {c["rule"] for c in out["checks"]} == set(PMKISAN["eligibility"])


def test_age_bounds(farmer):
    scheme = {"scheme_id": "X", "eligibility": {"age_min": 18, "age_max": 40}}
    assert check_scheme(scheme, farmer)["verdict"] == "ineligible"      # 42
    younger = farmer.model_copy(update={"age": 30})
    assert check_scheme(scheme, younger)["verdict"] == "eligible"


def test_any_of_alternatives(farmer):
    scheme = {
        "scheme_id": "SU",
        "eligibility": {"any_of": [{"social_category": ["sc", "st"]}, {"gender": ["female"]}]},
    }
    assert check_scheme(scheme, farmer)["verdict"] == "ineligible"      # obc male
    woman = farmer.model_copy(update={"gender": "female"})
    assert check_scheme(scheme, woman)["verdict"] == "eligible"
    st = farmer.model_copy(update={"social_category": "st"})
    assert check_scheme(scheme, st)["verdict"] == "eligible"


def test_income_falls_back_to_monthly_when_household_is_absent():
    profile = UserProfile(user_id="x", monthly_income=10_000, annual_household_income=None)
    scheme = {"scheme_id": "Y", "eligibility": {"annual_income_max": 200_000}}
    assert check_scheme(scheme, profile)["verdict"] == "eligible"       # 120k <= 200k


def test_residence_rule(farmer):
    urban_only = {"scheme_id": "U", "eligibility": {"residence": "urban"}}
    assert check_scheme(urban_only, farmer)["verdict"] == "ineligible"  # rural


def test_scheme_with_no_rules_is_eligible(farmer):
    assert check_scheme({"scheme_id": "OPEN", "eligibility": {}}, farmer)["verdict"] == "eligible"


# --------------------------------------------------------------------------- #
# Whole-catalogue assessment
# --------------------------------------------------------------------------- #

def test_farmer_qualifies_for_the_expected_flagship_schemes(farmer):
    out = eligibility_advisor(farmer)
    eligible_ids = {r["scheme_id"] for r in out["eligible"]}
    assert {"PM-KISAN", "PMFBY", "KCC"} <= eligible_ids


def test_verdict_buckets_partition_the_catalogue(farmer):
    out = eligibility_advisor(farmer)
    assert (
        len(out["eligible"]) + len(out["possibly_eligible"]) + len(out["ineligible"])
        == out["schemes_evaluated"]
    )


def test_taxpayer_loses_the_means_tested_schemes(farmer):
    rich = farmer.model_copy(update={
        "is_income_tax_payer": True, "annual_household_income": 2_500_000
    })
    eligible_ids = {r["scheme_id"] for r in eligibility_advisor(rich)["eligible"]}
    assert "PM-KISAN" not in eligible_ids
    assert "PMJAY" not in eligible_ids


def test_agent_reports_which_missing_field_would_unlock_most(farmer):
    out = eligibility_advisor(farmer)
    assert isinstance(out["ask_user_for"], list)


def test_empty_catalogue_is_handled(farmer):
    out = eligibility_advisor(farmer, schemes=[])
    assert out["schemes_evaluated"] == 0
    assert out["eligible"] == []


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(farmer):
    patch = eligibility_node(new_state(farmer))
    assert set(patch) == {"eligibility_result"}
    json.dumps(patch)


def test_node_survives_a_bare_profile(zero_profile):
    result = eligibility_node(new_state(zero_profile))["eligibility_result"]
    assert result["schemes_evaluated"] > 0
