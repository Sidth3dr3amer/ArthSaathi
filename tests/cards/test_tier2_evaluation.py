"""Credit Card Tier 2 -> Reward, Lounge, Membership and Cost agents."""

from __future__ import annotations

import json

import pytest

from ml.src.cards.tier1_profiler import run_tier1
from ml.src.cards.tier2_evaluation import (
    DOMESTIC_LOUNGE_VALUE,
    INTERNATIONAL_LOUNGE_VALUE,
    MEMBERSHIP_REALISATION,
    cost_agent_advisor,
    evaluate_all,
    evaluate_card,
    lounge_valuation_advisor,
    membership_valuation_advisor,
    reward_simulation_advisor,
)
from ml.src.councils.growth.credit_card import (
    analyze_spend_profile,
    profile_to_engine_dict,
)

CARD = {
    "card_name": "Test Card", "card_type": "TRAVEL", "card_tier": "Premium",
    "annual_fee": 5_000, "joining_fee": 0, "fee_waiver_spend": 500_000,
    "base_reward_rate": 2, "forex_markup": 1.5,
    "domestic_lounge_visits": 8, "international_lounge_visits": 4,
    "hotel_membership_value": 4_000, "movie_benefit_value": 1_200,
    "milestone_value_annual": 3_000,
}


# =========================================================================== #
# Reward Simulation
# =========================================================================== #

def test_rewards_are_delegated_to_the_tested_engine(engine_bits, tier1):
    engine_profile, spend_analysis, eligible, _ = engine_bits
    out = reward_simulation_advisor(eligible[0], engine_profile, spend_analysis, tier1["twin"])
    assert out["annual_rewards"] > 0
    assert out["components"]


def test_a_revolver_earns_no_real_rewards(engine_bits, revolver):
    """Rewards are discounted to zero when interest exceeds any reward rate."""
    engine_profile, spend_analysis, eligible, _ = engine_bits
    twin = run_tier1(revolver)["twin"]
    out = reward_simulation_advisor(eligible[0], engine_profile, spend_analysis, twin)
    assert out["annual_rewards"] == 0
    assert out["headline_rewards"] > 0
    assert out["realisation_factor"] == 0.0
    assert "revolves" in out["note"]


def test_no_twin_means_no_discount(engine_bits):
    engine_profile, spend_analysis, eligible, _ = engine_bits
    out = reward_simulation_advisor(eligible[0], engine_profile, spend_analysis, None)
    assert out["annual_rewards"] == out["headline_rewards"]


# =========================================================================== #
# Lounge Valuation
# =========================================================================== #

def test_lounge_visits_are_capped_by_realistic_travel():
    """The doc's formula is visits x value; the twin caps `visits`."""
    out = lounge_valuation_advisor(CARD, {"realistic_lounge_visits": 3})
    assert out["visits_offered"] == 12
    assert out["visits_used"] == 3
    assert out["visits_wasted"] == 9


def test_international_visits_are_allocated_first():
    """They are worth more, so a limited traveller should get the valuable ones."""
    out = lounge_valuation_advisor(CARD, {"realistic_lounge_visits": 2})
    assert out["international_used"] == 2
    assert out["domestic_used"] == 0
    assert out["lounge_value"] == 2 * INTERNATIONAL_LOUNGE_VALUE


def test_a_non_traveller_realises_nothing():
    out = lounge_valuation_advisor(CARD, {"realistic_lounge_visits": 0})
    assert out["lounge_value"] == 0
    assert out["utilisation"] == 0.0


def test_usage_cannot_exceed_what_is_offered():
    out = lounge_valuation_advisor(CARD, {"realistic_lounge_visits": 99})
    assert out["visits_used"] == 12
    assert out["visits_wasted"] == 0
    assert out["utilisation"] == 1.0


def test_a_card_with_no_lounge_access_is_zero():
    out = lounge_valuation_advisor({"domestic_lounge_visits": 0}, {"realistic_lounge_visits": 5})
    assert out["lounge_value"] == 0
    assert out["utilisation"] == 0.0


def test_domestic_value_is_used_when_no_international_visits():
    card = {"domestic_lounge_visits": 4, "international_lounge_visits": 0}
    out = lounge_valuation_advisor(card, {"realistic_lounge_visits": 2})
    assert out["lounge_value"] == 2 * DOMESTIC_LOUNGE_VALUE


# =========================================================================== #
# Membership Valuation
# =========================================================================== #

@pytest.mark.parametrize("profile,factor", sorted(MEMBERSHIP_REALISATION.items()))
def test_membership_realisation_scales_with_travel(profile, factor):
    out = membership_valuation_advisor(CARD, {"travel_profile": profile})
    hotel = next(i for i in out["items"] if i["benefit"] == "Hotel membership")
    assert hotel["realised_value"] == pytest.approx(4_000 * factor)


def test_face_value_and_realised_value_are_both_reported():
    """The gap between them is what card marketing omits."""
    out = membership_valuation_advisor(CARD, {"travel_profile": "low"})
    assert out["face_value"] > out["membership_value"]
    assert out["value_gap"] == pytest.approx(out["face_value"] - out["membership_value"])


def test_milestone_benefits_are_realised_in_full():
    out = membership_valuation_advisor(CARD, {"travel_profile": "low"})
    milestone = next(i for i in out["items"] if i["benefit"] == "Milestone benefit")
    assert milestone["realisation"] == 1.0


def test_a_card_with_no_memberships_is_zero():
    out = membership_valuation_advisor({"card_name": "Plain"}, {"travel_profile": "high"})
    assert out["membership_value"] == 0
    assert out["items"] == []


# =========================================================================== #
# Cost Agent
# =========================================================================== #

def test_cost_sums_fee_forex_and_interest_risk():
    twin = {"annual_spend": 100_000, "forex_exposure_annual": 60_000,
            "revolves_balance": True, "annual_interest_cost": 50_000}
    out = cost_agent_advisor(CARD, {}, twin)
    assert out["effective_fee"] == 5_000
    assert out["forex_cost"] == pytest.approx(60_000 * 0.015)
    assert out["interest_risk"] == pytest.approx(50_000 * 0.30)
    assert out["cost"] == pytest.approx(5_000 + 900 + 15_000)


def test_the_fee_waiver_applies_when_spend_clears_the_threshold():
    out = cost_agent_advisor(CARD, {}, {"annual_spend": 600_000})
    assert out["fee_waived"] is True
    assert out["effective_fee"] == 0


def test_the_shortfall_to_the_waiver_is_reported():
    out = cost_agent_advisor(CARD, {}, {"annual_spend": 400_000})
    assert out["fee_waived"] is False
    assert out["spend_shortfall_for_waiver"] == 100_000


def test_a_payer_carries_no_interest_risk():
    out = cost_agent_advisor(CARD, {}, {"annual_spend": 100_000, "revolves_balance": False})
    assert out["interest_risk"] == 0


def test_no_forex_spend_means_no_forex_cost():
    out = cost_agent_advisor(CARD, {}, {"annual_spend": 100_000})
    assert out["forex_cost"] == 0


# =========================================================================== #
# Combined evaluation
# =========================================================================== #

def test_net_is_gross_minus_cost(evaluations):
    for e in evaluations:
        assert e["net_annual_value"] == pytest.approx(
            e["gross_value"] - e["cost"]["cost"], abs=0.02
        )


def test_gross_is_the_sum_of_the_three_value_agents(evaluations):
    for e in evaluations:
        assert e["gross_value"] == pytest.approx(
            e["rewards"]["annual_rewards"]
            + e["lounge"]["lounge_value"]
            + e["membership"]["membership_value"],
            abs=0.02,
        )


def test_the_realisation_gap_is_reported(engine_bits, tier1):
    """Brochure value minus realised value -- unused lounge visits, mostly."""
    engine_profile, spend_analysis, eligible, _ = engine_bits
    atlas = next(c for c in eligible if "Atlas" in c["card_name"])
    out = evaluate_card(atlas, engine_profile, spend_analysis, tier1["twin"])
    assert out["headline_gross"] > out["gross_value"]
    assert out["realisation_gap"] > 0


def test_cards_are_ranked_by_net_value(evaluations):
    nets = [e["net_annual_value"] for e in evaluations]
    assert nets == sorted(nets, reverse=True)


def test_evaluation_is_json_serialisable(evaluations):
    json.dumps(evaluations)


def test_internal_breakdown_is_not_leaked(evaluations):
    for e in evaluations:
        assert "_breakdown" not in e["rewards"]


def test_evaluating_no_cards_returns_nothing(engine_bits, tier1):
    engine_profile, spend_analysis, _, _ = engine_bits
    assert evaluate_all([], engine_profile, spend_analysis, tier1["twin"]) == []
