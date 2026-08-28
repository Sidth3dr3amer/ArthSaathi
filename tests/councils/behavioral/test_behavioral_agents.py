"""
Behavioral Council -> Bias Detection, Habit Formation, Nudge Strategy, Literacy.

The four agents share a dataset and compose in sequence, so they are tested in
one file against the seeded history whose signals are known ground truth.
"""

from __future__ import annotations

import json

import pytest

from ml.src.common.synthetic import load_transactions
from ml.src.councils.behavioral.bias_detection import (
    bias_detection_advisor,
    bias_detection_node,
)
from ml.src.councils.behavioral.habit_formation import (
    habit_formation_advisor,
    habit_formation_node,
)
from ml.src.councils.behavioral.literacy import (
    CONCEPTS,
    MAX_LESSONS,
    literacy_advisor,
    literacy_node,
)
from ml.src.councils.behavioral.nudge_strategy import (
    MAX_ACTIVE_NUDGES,
    nudge_strategy_advisor,
    nudge_strategy_node,
)
from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture(scope="module")
def txns():
    return load_transactions()


@pytest.fixture(scope="module")
def bias(txns):
    return bias_detection_advisor(txns)


@pytest.fixture
def stressed_profile() -> UserProfile:
    return UserProfile(
        user_id="beh", age=31, job_type="salaried", monthly_income=90_000,
        essential_expenses=45_000, existing_emergency_fund=25_000,
        current_balance=60_000, dependents=2, has_health_insurance=False,
        debts=[
            Debt(name="HDFC Card", debt_type="credit_card",
                 outstanding_amount=150_000, interest_rate=42.0, minimum_due=7_500),
            Debt(name="Personal Loan", debt_type="personal_loan",
                 outstanding_amount=300_000, interest_rate=14.0, emi=9_000),
        ],
    )


# =========================================================================== #
# Bias Detection
# =========================================================================== #

def test_all_planted_biases_are_detected(bias):
    assert {
        "hyperbolic_discounting", "lifestyle_inflation",
        "status_quo_bias", "impulse_buying",
    } <= set(bias["biases_detected"])


def test_month_end_spike_is_detected_and_named(bias):
    spikes = [f for f in bias["findings"] if "last 5 days" in f["label"]]
    assert spikes
    assert any("dining" in f["evidence"]["category"] for f in spikes)


def test_every_finding_carries_evidence_and_a_price(bias):
    for finding in bias["findings"]:
        assert finding["evidence"]
        assert finding["estimated_annual_cost"] >= 0
        assert 0.0 <= finding["strength"] <= 1.0


def test_findings_lead_with_an_observation_not_a_label(bias):
    """'You have present bias' is an insult; a number is information."""
    for finding in bias["findings"]:
        assert len(finding["observation"]) > 40
        assert any(ch.isdigit() for ch in finding["observation"])


def test_findings_are_ranked_by_cost(bias):
    costs = [f["estimated_annual_cost"] for f in bias["findings"]]
    assert costs == sorted(costs, reverse=True)


def test_no_history_is_handled():
    out = bias_detection_advisor([])
    assert out["status"] == "no transaction history"
    assert out["findings"] == []


def test_clean_history_produces_no_findings():
    """Steady, mid-month, well-saved spending must not be pathologised."""
    txns = []
    for month in range(1, 13):
        txns.append({"date": f"2026-{month:02d}-01", "amount": 100_000,
                     "category": "income", "merchant": "Salary",
                     "direction": "credit", "is_recurring": True})
        for day in (12, 15, 18):
            txns.append({"date": f"2026-{month:02d}-{day}", "amount": 2_000,
                         "category": "dining", "merchant": "Cafe",
                         "direction": "debit", "is_recurring": False})
        txns.append({"date": f"2026-{month:02d}-05", "amount": 40_000,
                     "category": "savings", "merchant": "SIP",
                     "direction": "debit", "is_recurring": True})
    out = bias_detection_advisor(txns)
    assert out["findings"] == []
    assert out["status"] == "No material behavioural patterns detected"


def test_bias_node_contract(salaried_profile):
    patch = bias_detection_node(new_state(salaried_profile))
    assert set(patch) == {"bias_detection_result"}
    json.dumps(patch)


def test_bias_node_prefers_transactions_from_state(salaried_profile):
    state = new_state(salaried_profile)
    state["transactions"] = []
    assert bias_detection_node(state)["bias_detection_result"]["months_analysed"] == 0


# =========================================================================== #
# Habit Formation
# =========================================================================== #

def test_habits_are_written_as_implementation_intentions(txns, bias):
    out = habit_formation_advisor(txns, bias["findings"])
    assert out["proposed_habits"]
    for habit in out["proposed_habits"]:
        assert habit["implementation_intention"].startswith("When ")
        assert habit["cue"] and habit["action"]


def test_exactly_one_keystone_is_nominated(txns, bias):
    out = habit_formation_advisor(txns, bias["findings"])
    assert out["keystone_habit"] is not None
    assert out["keystone_habit"] == out["proposed_habits"][0]


def test_keystone_is_chosen_by_leverage_not_raw_value(txns, bias):
    out = habit_formation_advisor(txns, bias["findings"])
    leverages = [h["leverage"] for h in out["proposed_habits"]]
    assert leverages == sorted(leverages, reverse=True)


def test_habits_anchor_to_observed_routines(txns, bias):
    out = habit_formation_advisor(txns, bias["findings"])
    assert out["strong_routines"]
    assert any(h["anchor"] for h in out["proposed_habits"])


def test_no_bias_findings_means_no_proposals(txns):
    out = habit_formation_advisor(txns, bias_findings=[])
    assert out["proposed_habits"] == []
    assert out["keystone_habit"] is None


def test_habit_node_consumes_bias_upstream(stressed_profile):
    state = new_state(stressed_profile, query="habits")
    state.update(bias_detection_node(state))
    patch = habit_formation_node(state)
    assert set(patch) == {"habit_formation_result"}
    assert patch["habit_formation_result"]["keystone_habit"]


# =========================================================================== #
# Nudge Strategy
# =========================================================================== #

def test_active_nudges_are_capped(bias):
    out = nudge_strategy_advisor(bias["findings"], monthly_surplus=20_000)
    assert out["active_count"] <= MAX_ACTIVE_NUDGES
    assert out["suppressed_count"] >= 0


def test_the_cap_is_explained(bias):
    out = nudge_strategy_advisor(bias["findings"])
    assert "attention" in out["cap_rationale"]


def test_active_nudges_never_share_a_trigger(bias, txns):
    """Two prompts on the same moment is one nudge and one annoyance."""
    habits = habit_formation_advisor(txns, bias["findings"])
    out = nudge_strategy_advisor(bias["findings"], habits=habits, monthly_surplus=20_000)
    triggers = [n["trigger"] for n in out["active_nudges"]]
    assert len(triggers) == len(set(triggers))


def test_every_nudge_names_its_mechanism(bias):
    out = nudge_strategy_advisor(bias["findings"])
    for nudge in out["active_nudges"] + out["queued_nudges"]:
        assert nudge["mechanism"]
        assert nudge["mechanism_description"]
        assert nudge["timing"]


def test_nudges_are_ranked_by_expected_value(bias):
    out = nudge_strategy_advisor(bias["findings"])
    values = [n["expected_annual_value"] for n in out["active_nudges"]]
    assert values == sorted(values, reverse=True)


def test_no_findings_means_no_nudges():
    out = nudge_strategy_advisor([])
    assert out["active_nudges"] == []
    assert out["status"] == "No nudges indicated"


def test_critical_runway_adds_a_framing_nudge():
    out = nudge_strategy_advisor([], emergency_status="Critical")
    assert any(n["mechanism"] == "framing" for n in out["active_nudges"])


def test_nudge_node_composes_with_upstream(stressed_profile):
    state = new_state(stressed_profile, query="nudge me")
    state.update(emergency_fund_node(state))
    state.update(bias_detection_node(state))
    state.update(habit_formation_node(state))
    patch = nudge_strategy_node(state)
    assert set(patch) == {"nudge_strategy_result"}
    json.dumps(patch)


# =========================================================================== #
# Financial Literacy
# =========================================================================== #

def test_expensive_debt_triggers_the_compounding_lesson(stressed_profile):
    out = literacy_advisor(stressed_profile)
    assert any(l["concept"] == "compound_interest_on_debt" for l in out["curriculum"])


def test_lessons_are_personalised_with_the_users_own_numbers(stressed_profile):
    out = literacy_advisor(stressed_profile)
    lesson = next(l for l in out["curriculum"]
                  if l["concept"] == "compound_interest_on_debt")
    assert "150,000" in lesson["personalised"]
    assert "42%" in lesson["personalised"]


def test_missing_health_cover_triggers_its_lesson(stressed_profile):
    out = literacy_advisor(stressed_profile)
    concepts = {l["concept"] for l in out["curriculum"]} | set(out["deferred"])
    assert "health_insurance" in concepts


def test_curriculum_is_capped(stressed_profile, bias):
    out = literacy_advisor(stressed_profile, bias["findings"], emergency_status="Critical")
    assert len(out["curriculum"]) <= MAX_LESSONS
    assert out["gaps_identified"] >= len(out["curriculum"])


def test_curriculum_is_ordered_by_cost_of_the_gap(stressed_profile, bias):
    out = literacy_advisor(stressed_profile, bias["findings"])
    costs = [l["estimated_annual_cost_of_gap"] for l in out["curriculum"]]
    assert costs == sorted(costs, reverse=True)


def test_a_well_managed_profile_has_few_gaps():
    tidy = UserProfile(
        user_id="tidy", age=35, monthly_income=200_000, essential_expenses=60_000,
        existing_emergency_fund=600_000, current_balance=100_000,
        has_health_insurance=True, has_term_cover=True, dependents=0,
    )
    out = literacy_advisor(tidy)
    assert out["gaps_identified"] <= 2


def test_every_concept_has_a_plain_language_body():
    for concept, body in CONCEPTS.items():
        assert body["title"] and body["plain"] and body["why_it_matters"]
        assert len(body["plain"]) > 80


def test_literacy_node_composes_with_upstream(stressed_profile):
    state = new_state(stressed_profile, query="teach me")
    state.update(emergency_fund_node(state))
    state.update(bias_detection_node(state))
    patch = literacy_node(state)
    assert set(patch) == {"literacy_result"}
    json.dumps(patch)


def test_literacy_node_survives_a_bare_profile(zero_profile):
    result = literacy_node(new_state(zero_profile))["literacy_result"]
    assert result["literacy_level"]
