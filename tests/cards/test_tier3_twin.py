"""Credit Card Tier 3 -> 12-month Financial Twin simulation."""

from __future__ import annotations

import json

import pytest

from ml.src.cards.tier3_twin import SCENARIOS, simulate_all, simulate_card

EVALUATION = {
    "card_name": "Test Card",
    "rewards": {"annual_rewards": 12_000.0},
    "lounge": {"visits_used": 8, "international_used": 2, "visits_wasted": 4,
               "domestic_used": 6, "lounge_value": 10_200.0},
    "membership": {"membership_value": 4_000.0},
    "cost": {"annual_fee": 5_000.0, "joining_fee": 0.0, "forex_cost": 900.0,
             "interest_risk": 0.0, "fee_waiver_spend": 500_000.0},
}
TWIN = {"annual_spend": 450_000.0}


def test_three_named_scenarios_are_produced():
    out = simulate_card(EVALUATION, TWIN)
    assert set(out["scenarios"]) == set(SCENARIOS)
    assert {"best", "avg", "worst"} <= set(out)


def test_scenarios_are_ordered_best_above_average_above_worst():
    out = simulate_card(EVALUATION, TWIN)
    assert out["best"] > out["avg"] > out["worst"]


def test_spread_is_the_gap_between_best_and_worst():
    out = simulate_card(EVALUATION, TWIN)
    assert out["spread"] == pytest.approx(out["best"] - out["worst"], abs=0.02)


def test_rewards_scale_with_spend():
    out = simulate_card(EVALUATION, TWIN)
    best_mult = SCENARIOS["best"][0]
    assert out["scenarios"]["best"]["rewards"] == pytest.approx(12_000 * best_mult)


def test_lounge_use_falls_in_a_bad_year():
    out = simulate_card(EVALUATION, TWIN)
    assert out["scenarios"]["worst"]["lounge_visits"] < out["scenarios"]["best"]["lounge_visits"]


def test_a_spend_linked_waiver_flips_between_scenarios():
    """Spending 20% more clears the threshold; 20% less misses it."""
    twin = {"annual_spend": 450_000.0}          # x1.2 = 540k clears, x0.8 = 360k misses
    out = simulate_card(EVALUATION, twin)
    assert out["scenarios"]["best"]["fee_waived"] is True
    assert out["scenarios"]["best"]["fee_paid"] == 0
    assert out["scenarios"]["worst"]["fee_waived"] is False
    assert out["scenarios"]["worst"]["fee_paid"] == 5_000


def test_a_card_with_no_waiver_always_charges_the_fee():
    evaluation = {**EVALUATION, "cost": {**EVALUATION["cost"], "fee_waiver_spend": 0}}
    out = simulate_card(evaluation, TWIN)
    assert all(s["fee_paid"] == 5_000 for s in out["scenarios"].values())


def test_volatility_is_bounded():
    out = simulate_card(EVALUATION, TWIN)
    assert 0.0 <= out["volatility"] <= 1.0


def test_a_steady_card_is_less_volatile_than_a_perk_heavy_one():
    steady = {**EVALUATION,
              "lounge": {**EVALUATION["lounge"], "visits_used": 0, "international_used": 0,
                         "domestic_used": 0, "lounge_value": 0.0},
              "membership": {"membership_value": 0.0},
              "cost": {**EVALUATION["cost"], "fee_waiver_spend": 0}}
    assert simulate_card(steady, TWIN)["volatility"] < simulate_card(EVALUATION, TWIN)["volatility"]


def test_a_negative_downside_is_flagged():
    expensive = {**EVALUATION,
                 "rewards": {"annual_rewards": 500.0},
                 "lounge": {**EVALUATION["lounge"], "visits_used": 0, "international_used": 0,
                            "domestic_used": 0, "lounge_value": 0.0},
                 "membership": {"membership_value": 0.0},
                 "cost": {**EVALUATION["cost"], "annual_fee": 20_000.0, "fee_waiver_spend": 0}}
    out = simulate_card(expensive, TWIN)
    assert out["downside_is_negative"] is True
    assert out["worst"] < 0


def test_a_shorter_horizon_scales_the_result_down():
    year = simulate_card(EVALUATION, TWIN, months=12)
    half = simulate_card(EVALUATION, TWIN, months=6)
    assert half["avg"] < year["avg"]


def test_no_twin_still_simulates():
    out = simulate_card(EVALUATION, None)
    assert out["best"] > out["worst"]


def test_zero_average_does_not_divide_by_zero():
    flat = {**EVALUATION,
            "rewards": {"annual_rewards": 0.0},
            "lounge": {**EVALUATION["lounge"], "visits_used": 0, "international_used": 0,
                       "domestic_used": 0, "lounge_value": 0.0},
            "membership": {"membership_value": 0.0},
            "cost": {"annual_fee": 0.0, "joining_fee": 0.0, "forex_cost": 0.0,
                     "interest_risk": 0.0, "fee_waiver_spend": 0.0}}
    out = simulate_card(flat, {"annual_spend": 0})
    assert out["volatility"] in (0.0, 1.0)


def test_simulate_all_covers_every_card(evaluations, tier1):
    out = simulate_all(evaluations, tier1["twin"])
    assert len(out) == len(evaluations)
    assert {s["card_name"] for s in out} == {e["card_name"] for e in evaluations}


def test_simulation_is_json_serialisable(simulations):
    json.dumps(simulations)


def test_simulation_is_deterministic():
    assert simulate_card(EVALUATION, TWIN) == simulate_card(EVALUATION, TWIN)
