"""Benefits Council -> Scheme Matching Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.benefits.scheme_matching import (
    REALISATION,
    _annual_value,
    scheme_matching_advisor,
    scheme_matching_node,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture
def farmer() -> UserProfile:
    return UserProfile(
        user_id="rahul", name="Rahul Patil", age=42, job_type="business",
        occupation="farmer", residence="rural", gender="male",
        monthly_income=35_000, essential_expenses=22_000,
        annual_household_income=420_000, land_holding_ha=1.5, dependents=3,
        has_health_insurance=False,
    )


@pytest.fixture
def urban_worker() -> UserProfile:
    return UserProfile(
        user_id="w", age=26, job_type="unsalaried", occupation="street_vendor",
        residence="urban", gender="female", social_category="sc",
        monthly_income=12_000, essential_expenses=10_000,
        annual_household_income=144_000, has_health_insurance=False,
    )


# --------------------------------------------------------------------------- #
# Annual value — the contingent-vs-guaranteed correction
# --------------------------------------------------------------------------- #

def test_insurance_is_not_valued_at_its_sum_assured():
    """
    A Rs 20/year accident policy is not a Rs 2,00,000/year benefit. Valuing the
    sum assured directly would rank PMSBY above a guaranteed cash transfer.
    """
    value = _annual_value({"type": "insurance", "amount": 200_000, "frequency": "annual"})
    assert value < 20_000
    assert value == pytest.approx(200_000 * REALISATION["insurance"])


def test_guaranteed_cash_transfer_is_valued_in_full():
    value = _annual_value({"type": "cash_transfer", "amount": 6_000, "frequency": "annual"})
    assert value == pytest.approx(6_000)


def test_a_guaranteed_transfer_outvalues_a_far_larger_credit_line():
    transfer = _annual_value({"type": "cash_transfer", "amount": 6_000, "frequency": "annual"})
    credit = _annual_value({"type": "credit_line", "amount": 300_000, "frequency": "revolving"})
    assert credit < transfer * 3        # not 50x, as the raw amounts would imply


def test_one_time_benefits_are_amortised():
    value = _annual_value({"type": "subsidy", "amount": 120_000, "frequency": "one_time"})
    assert value == pytest.approx(120_000 / 5 * REALISATION["subsidy"])


def test_zero_amount_is_zero_value():
    assert _annual_value({"type": "cash_transfer", "amount": 0}) == 0.0


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def test_matches_are_sorted_by_score(farmer):
    out = scheme_matching_advisor(farmer)
    scores = [m["match_score"] for m in out["matches"]]
    assert scores == sorted(scores, reverse=True)


def test_occupation_targeted_schemes_rank_for_that_occupation(farmer):
    """PM-KISAN is the canonical recommendation for a smallholder."""
    out = scheme_matching_advisor(farmer, top_n=5)
    top_ids = [m["scheme_id"] for m in out["matches"]]
    assert "PM-KISAN" in top_ids


def test_targeting_is_cited_in_the_reasons(farmer):
    out = scheme_matching_advisor(farmer)
    kisan = next(m for m in out["all_scored"] if m["scheme_id"] == "PM-KISAN")
    assert any("targeted at farmer" in w for w in kisan["why"])


def test_crop_insurance_is_not_justified_by_dependants(farmer):
    """PMFBY is crop cover; it must not be recommended for having a family."""
    out = scheme_matching_advisor(farmer)
    pmfby = next(m for m in out["all_scored"] if m["scheme_id"] == "PMFBY")
    assert not any("life cover" in w for w in pmfby["why"])
    assert any("crop" in w for w in pmfby["why"])


def test_uninsured_low_income_user_is_matched_to_health_first(urban_worker):
    out = scheme_matching_advisor(urban_worker, top_n=3)
    assert out["matches"][0]["scheme_id"] == "PMJAY"
    assert any("health" in w for w in out["matches"][0]["why"])


def test_scores_are_bounded(farmer):
    for match in scheme_matching_advisor(farmer)["all_scored"]:
        assert 0 <= match["match_score"] <= 100


def test_unconfirmed_eligibility_is_discounted():
    """A scheme we are unsure about must rank below an equivalent confirmed one."""
    profile = UserProfile(user_id="p", age=30, monthly_income=20_000, occupation="farmer")
    unknown_land = profile.model_copy(update={"land_holding_ha": None})
    known_land = profile.model_copy(update={"land_holding_ha": 1.0, "has_bank_account": True})

    def kisan_score(p):
        rows = scheme_matching_advisor(p)["all_scored"]
        row = next((r for r in rows if r["scheme_id"] == "PM-KISAN"), None)
        return row["match_score"] if row else 0

    assert kisan_score(unknown_land) < kisan_score(known_land)


def test_top_n_is_respected(farmer):
    assert len(scheme_matching_advisor(farmer, top_n=2)["matches"]) == 2


def test_excluding_possibles_narrows_the_candidate_set(farmer):
    with_possible = scheme_matching_advisor(farmer, top_n=99, include_possible=True)
    without = scheme_matching_advisor(farmer, top_n=99, include_possible=False)
    assert len(without["all_scored"]) <= len(with_possible["all_scored"])
    assert all(r["verdict"] == "eligible" for r in without["all_scored"])


def test_estimated_benefit_counts_only_confirmed_schemes(farmer):
    out = scheme_matching_advisor(farmer)
    expected = sum(r["annual_value"] for r in out["all_scored"] if r["verdict"] == "eligible")
    assert out["estimated_annual_benefit"] == pytest.approx(expected, abs=0.5)


def test_every_match_carries_a_reason(farmer):
    for match in scheme_matching_advisor(farmer)["matches"]:
        assert match["why"]


def test_empty_catalogue_yields_no_matches(farmer):
    out = scheme_matching_advisor(farmer, schemes=[])
    assert out["matches"] == []
    assert out["estimated_annual_benefit"] == 0


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(farmer):
    patch = scheme_matching_node(new_state(farmer))
    assert set(patch) == {"scheme_matching_result"}
    json.dumps(patch)


def test_node_produces_human_readable_recommendations(farmer):
    result = scheme_matching_node(new_state(farmer))["scheme_matching_result"]
    assert result["recommendations"]
    assert all("% match" in r for r in result["recommendations"])


def test_node_survives_a_bare_profile(zero_profile):
    result = scheme_matching_node(new_state(zero_profile))["scheme_matching_result"]
    assert result["schemes_evaluated"] > 0
