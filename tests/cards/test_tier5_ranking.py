"""
Credit Card Tier 5 -> the weighted Ranking Engine.

The weights come from `docs/CreditCardAgentWorking.md` and are pinned here, so a
change to the formula has to be deliberate rather than incidental.
"""

from __future__ import annotations

import json

import pytest

from ml.src.cards.tier5_ranking import (
    WEIGHTS,
    approval_probability_score,
    future_value_score,
    rank_cards,
    user_match_score,
)


@pytest.fixture
def ranking(evaluations, simulations, deliberation, engine_bits, tier1):
    _, _, eligible, _ = engine_bits
    return rank_cards(evaluations, simulations, deliberation, eligible,
                      tier1["profiler"], tier1["spending"], tier1["twin"])


# --------------------------------------------------------------------------- #
# The formula
# --------------------------------------------------------------------------- #

def test_weights_match_the_design_doc():
    assert WEIGHTS == {
        "net_annual_value": 0.35,
        "user_match": 0.20,
        "approval_probability": 0.15,
        "future_value": 0.15,
        "agent_consensus": 0.15,
    }


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_final_score_is_the_weighted_sum_of_its_components(ranking):
    for row in ranking["all_ranked"]:
        expected = sum(row["components"][k] * WEIGHTS[k] for k in WEIGHTS)
        assert row["final_score"] == pytest.approx(expected, abs=1e-4)


def test_weighted_contributions_reconstruct_the_score(ranking):
    for row in ranking["all_ranked"]:
        assert sum(row["weighted_contributions"].values()) == pytest.approx(
            row["final_score"], abs=1e-3
        )


def test_every_component_is_normalised_to_a_unit_scale(ranking):
    for row in ranking["all_ranked"]:
        for name, value in row["components"].items():
            assert 0.0 <= value <= 1.0, f"{row['card_name']}.{name} = {value}"


def test_final_scores_are_bounded(ranking):
    for row in ranking["all_ranked"]:
        assert 0.0 <= row["final_score"] <= 1.0


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #

def test_net_value_is_normalised_against_this_comparison(ranking):
    """The best card in the set scores 1.0, the worst 0.0."""
    nets = [r["components"]["net_annual_value"] for r in ranking["all_ranked"]]
    assert max(nets) == 1.0
    assert min(nets) == 0.0


def test_user_match_rewards_a_matching_card_type():
    spending = {"buckets_annual": {"online": 100_000, "grocery": 20_000},
                "dominant_category": "online"}
    cashback = user_match_score({}, {"card_type": "CASHBACK"}, spending)
    travel = user_match_score({}, {"card_type": "TRAVEL"}, spending)
    assert cashback["score"] > travel["score"]
    assert cashback["dominant_matched"] is True


def test_user_match_handles_zero_spend():
    out = user_match_score({}, {"card_type": "CASHBACK"}, {"buckets_annual": {}})
    assert out["score"] == 0.0


def test_approval_rises_with_income_headroom():
    profiler = {"income": 1_200_000, "salaried": True}
    twin = {"approval_headroom": True}
    comfortable = approval_probability_score(
        {"income_requirement": 300_000}, profiler, twin)
    marginal = approval_probability_score(
        {"income_requirement": 1_500_000}, profiler, twin)
    assert comfortable["score"] > marginal["score"]


def test_an_invite_only_card_is_near_unobtainable():
    out = approval_probability_score(
        {"invite_only": True}, {"income": 5_000_000, "salaried": True},
        {"approval_headroom": True})
    assert out["score"] <= 0.2
    assert "invite-only" in out["reason"]


def test_high_utilisation_reduces_approval_odds():
    card = {"income_requirement": 300_000}
    profiler = {"income": 1_200_000, "salaried": True}
    clean = approval_probability_score(card, profiler, {"approval_headroom": True})
    stretched = approval_probability_score(card, profiler, {"approval_headroom": False})
    assert stretched["score"] < clean["score"]
    assert "utilisation" in stretched["reason"]


def test_future_value_is_built_on_the_downside_not_the_average():
    steady = future_value_score({"worst": 10_000, "volatility": 0.1}, 10_000, 0)
    swingy = future_value_score({"worst": 10_000, "volatility": 0.9}, 10_000, 0)
    assert steady["score"] > swingy["score"]
    assert swingy["volatility_penalty"] > steady["volatility_penalty"]


def test_future_value_handles_a_single_card():
    out = future_value_score({"worst": 5_000, "volatility": 0.2}, 5_000, 5_000)
    assert 0.0 <= out["score"] <= 1.0


# --------------------------------------------------------------------------- #
# Ranking output
# --------------------------------------------------------------------------- #

def test_cards_are_ordered_by_final_score(ranking):
    scores = [r["final_score"] for r in ranking["all_ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_the_winner_is_the_top_ranked_card(ranking):
    assert ranking["winner"] == ranking["all_ranked"][0]["card_name"]


def test_the_margin_over_the_runner_up_is_reported(ranking):
    assert ranking["runner_up"] is not None
    assert ranking["margin_over_runner_up"] == pytest.approx(
        ranking["all_ranked"][0]["net_annual_value"]
        - ranking["all_ranked"][1]["net_annual_value"], abs=0.02
    )


def test_the_decisive_component_is_the_largest_contribution(ranking):
    top = ranking["all_ranked"][0]
    assert ranking["decisive_component"] == max(
        top["weighted_contributions"], key=lambda k: top["weighted_contributions"][k]
    )


def test_top_n_is_respected(evaluations, simulations, deliberation, engine_bits, tier1):
    _, _, eligible, _ = engine_bits
    out = rank_cards(evaluations, simulations, deliberation, eligible,
                     tier1["profiler"], tier1["spending"], tier1["twin"], top_n=2)
    assert len(out["ranked"]) == 2
    assert len(out["all_ranked"]) == len(evaluations)


def test_no_cards_degrades_rather_than_raising(tier1):
    out = rank_cards([], [], {}, [], tier1["profiler"], tier1["spending"], tier1["twin"])
    assert out["status"] == "no cards to rank"
    assert out["ranked"] == []


def test_ranking_is_json_serialisable(ranking):
    json.dumps(ranking)


def test_ranking_is_deterministic(evaluations, simulations, deliberation, engine_bits, tier1):
    _, _, eligible, _ = engine_bits
    args = (evaluations, simulations, deliberation, eligible,
            tier1["profiler"], tier1["spending"], tier1["twin"])
    assert rank_cards(*args) == rank_cards(*args)
