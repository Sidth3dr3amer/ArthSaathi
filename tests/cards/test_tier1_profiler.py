"""Credit Card Tier 1 -> User Profiler, Spending Analyzer, Financial Twin."""

from __future__ import annotations

import json

import pytest

from ml.src.cards.tier1_profiler import (
    CATEGORY_MAP,
    REVOLVING_RATE_THRESHOLD,
    TYPICALLY_EXCLUDED,
    financial_twin_advisor,
    run_tier1,
    spending_analyzer_advisor,
    tier1_node,
    user_profiler_advisor,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


def profiler(**overrides):
    base = dict(age=32, monthly_income=100_000, job_type="salaried",
                travel_frequency="occasional")
    base.update(overrides)
    return user_profiler_advisor(**base)


# =========================================================================== #
# User Profiler
# =========================================================================== #

def test_income_is_annualised():
    assert profiler(monthly_income=100_000)["income"] == 1_200_000


@pytest.mark.parametrize(
    "annual,band",
    [(200_000, "Entry"), (500_000, "Beginner"), (1_000_000, "Mid-range"),
     (2_000_000, "Premium"), (5_000_000, "Super-premium")],
)
def test_income_bands(annual, band):
    assert profiler(monthly_income=annual / 12)["income_band"] == band


@pytest.mark.parametrize(
    "frequency,expected",
    [("none", "low"), ("occasional", "medium"), ("frequent", "high")],
)
def test_travel_profile_from_stated_frequency(frequency, expected):
    assert profiler(travel_frequency=frequency)["travel_profile"] == expected


def test_forex_spend_lifts_the_travel_profile():
    """Someone spending heavily abroad travels, whatever they told us."""
    assert profiler(travel_frequency="none",
                    international_spend_monthly=20_000)["travel_profile"] == "high"


@pytest.mark.parametrize(
    "age,stage",
    [(22, "early_career"), (30, "establishing"), (40, "peak_earning"), (55, "pre_retirement")],
)
def test_life_stage(age, stage):
    assert profiler(age=age)["life_stage"] == stage


def test_govt_counts_as_salaried():
    assert profiler(job_type="govt")["salaried"] is True
    assert profiler(job_type="business")["salaried"] is False


# =========================================================================== #
# Spending Analyzer
# =========================================================================== #

def test_categories_map_into_card_reward_buckets():
    out = spending_analyzer_advisor({"online_shopping": 5_000, "groceries": 3_000})
    assert out["buckets_monthly"] == {"online": 5_000, "grocery": 3_000}


def test_multiple_categories_merge_into_one_bucket():
    out = spending_analyzer_advisor({"dining": 4_000, "entertainment": 2_000})
    assert out["buckets_monthly"]["dining"] == 6_000


def test_spend_is_annualised():
    out = spending_analyzer_advisor({"dining": 1_000})
    assert out["total_annual"] == 12_000


def test_rent_and_utilities_are_excluded_from_rewardable_spend():
    """Most cards exclude these, so counting them overstates every reward."""
    out = spending_analyzer_advisor({"dining": 5_000, "rent": 25_000, "utility_bills": 2_000})
    assert out["rewardable_monthly"] == 5_000
    assert out["excluded_share"] > 0.8


def test_unknown_categories_fall_into_other_and_are_reported():
    out = spending_analyzer_advisor({"crypto": 5_000})
    assert out["buckets_monthly"]["other"] == 5_000
    assert out["unmapped_categories"] == ["crypto"]


def test_dominant_category_is_the_largest():
    out = spending_analyzer_advisor({"dining": 2_000, "online_shopping": 9_000})
    assert out["dominant_category"] == "online"


def test_observed_transactions_override_stated_spend():
    txns = [
        {"date": "2026-01-05", "amount": 3_000, "category": "dining", "direction": "debit"},
        {"date": "2026-02-05", "amount": 3_000, "category": "dining", "direction": "debit"},
        {"date": "2026-01-10", "amount": 50_000, "category": "income", "direction": "credit"},
    ]
    out = spending_analyzer_advisor({"dining": 99_999}, transactions=txns)
    assert out["buckets_monthly"]["dining"] == 3_000
    assert "observed over 2 months" in out["source"]


def test_income_and_savings_rows_are_not_treated_as_spend():
    txns = [
        {"date": "2026-01-05", "amount": 1_000, "category": "dining", "direction": "debit"},
        {"date": "2026-01-06", "amount": 9_000, "category": "savings", "direction": "debit"},
    ]
    out = spending_analyzer_advisor({}, transactions=txns)
    assert set(out["buckets_monthly"]) == {"dining"}


def test_empty_spend_does_not_divide_by_zero():
    out = spending_analyzer_advisor({})
    assert out["total_monthly"] == 0
    assert out["excluded_share"] == 0.0
    assert out["dominant_category"] is None


def test_every_mapped_bucket_is_a_known_name():
    assert TYPICALLY_EXCLUDED <= set(CATEGORY_MAP.values())


# =========================================================================== #
# Financial Twin
# =========================================================================== #

def test_a_payer_has_real_rewards():
    twin = financial_twin_advisor(profiler(), spending_analyzer_advisor({"dining": 5_000}))
    assert twin["rewards_are_real"] is True
    assert twin["revolves_balance"] is False


def test_a_revolver_has_no_real_rewards():
    """Rewards earned while paying 42% interest are a rounding error on a loss."""
    twin = financial_twin_advisor(
        profiler(), spending_analyzer_advisor({"dining": 5_000}),
        monthly_surplus=5_000, revolving_debt=180_000, highest_debt_rate=42.0,
    )
    assert twin["revolves_balance"] is True
    assert twin["rewards_are_real"] is False
    assert twin["annual_interest_cost"] == pytest.approx(75_600)
    assert any("exceeds what any reward rate returns" in w for w in twin["warnings"])


def test_low_rate_debt_is_not_treated_as_revolving():
    twin = financial_twin_advisor(
        profiler(), spending_analyzer_advisor({}),
        revolving_debt=180_000, highest_debt_rate=REVOLVING_RATE_THRESHOLD - 1,
    )
    assert twin["revolves_balance"] is False


def test_lounge_visits_are_capped_by_actual_travel():
    """A card offering 30 visits is worthless to someone who flies twice."""
    rare = financial_twin_advisor(
        profiler(travel_frequency="none"), spending_analyzer_advisor({}))
    often = financial_twin_advisor(
        profiler(travel_frequency="frequent"), spending_analyzer_advisor({}))
    assert rare["realistic_lounge_visits"] == 0
    assert often["realistic_lounge_visits"] > rare["realistic_lounge_visits"]


def test_no_surplus_means_no_fee_tolerance():
    twin = financial_twin_advisor(
        profiler(), spending_analyzer_advisor({}), monthly_surplus=0)
    assert twin["fee_tolerance"] == 0
    assert twin["can_absorb_premium_fee"] is False
    assert any("funded from savings" in w for w in twin["warnings"])


def test_fee_tolerance_is_bounded_by_both_income_and_surplus():
    thin = financial_twin_advisor(
        profiler(monthly_income=100_000), spending_analyzer_advisor({}),
        monthly_surplus=1_000)
    assert thin["fee_tolerance"] == 3_000        # surplus x 3, below the income cap


def test_high_utilisation_removes_approval_headroom():
    twin = financial_twin_advisor(
        profiler(), spending_analyzer_advisor({}),
        monthly_surplus=20_000, credit_utilisation=0.7)
    assert twin["approval_headroom"] is False
    assert any("utilisation" in w for w in twin["warnings"])


def test_a_clean_profile_raises_no_warnings():
    twin = financial_twin_advisor(
        profiler(), spending_analyzer_advisor({"dining": 5_000}), monthly_surplus=30_000)
    assert twin["warnings"] == []


# =========================================================================== #
# Tier entry point
# =========================================================================== #

def test_run_tier1_returns_all_three_stages(card_user):
    out = run_tier1(card_user)
    assert set(out) == {"profiler", "spending", "twin"}


def test_run_tier1_wires_debt_into_the_twin(revolver):
    out = run_tier1(revolver)
    assert out["twin"]["revolves_balance"] is True
    assert out["twin"]["existing_cards"] == 1


def test_node_writes_one_key_and_serialises(card_user):
    patch = tier1_node(new_state(card_user))
    assert set(patch) == {"card_tier1_result"}
    json.dumps(patch)


def test_node_reads_transactions_from_state(card_user):
    state = new_state(card_user)
    state["transactions"] = [
        {"date": "2026-01-05", "amount": 1_000, "category": "dining", "direction": "debit"}
    ]
    result = tier1_node(state)["card_tier1_result"]
    assert "observed" in result["spending"]["source"]


def test_tier1_survives_a_bare_profile():
    out = run_tier1(UserProfile(user_id="bare"))
    assert out["twin"]["fee_tolerance"] == 0
    json.dumps(out)


def test_tier1_does_not_mutate_the_profile(card_user):
    before = card_user.model_dump()
    run_tier1(card_user)
    assert card_user.model_dump() == before
