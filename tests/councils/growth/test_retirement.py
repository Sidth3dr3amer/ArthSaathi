"""Growth Council -> Retirement Planning Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.growth.retirement import (
    EXPENSE_REPLACEMENT_RATIO,
    _fv_of_sip,
    _future_value,
    _sip_for_target,
    retirement_advisor,
    retirement_node,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


def advise(**overrides):
    base = dict(age=32, monthly_expenses=55_000, current_corpus=0.0,
                current_monthly_contribution=0.0)
    base.update(overrides)
    return retirement_advisor(**base)


# --------------------------------------------------------------------------- #
# Time-value helpers
# --------------------------------------------------------------------------- #

def test_future_value_compounds():
    assert _future_value(100_000, 10.0, 10) == pytest.approx(259_374, abs=5)


def test_sip_future_value_exceeds_the_sum_of_contributions():
    monthly, years = 10_000, 10
    assert _fv_of_sip(monthly, 11.0, years) > monthly * 12 * years


def test_sip_for_target_inverts_its_future_value():
    target = _fv_of_sip(15_000, 11.0, 20)
    assert _sip_for_target(target, 11.0, 20) == pytest.approx(15_000, rel=1e-6)


def test_zero_rate_sip_is_linear():
    assert _fv_of_sip(1_000, 0.0, 2) == pytest.approx(24_000)


def test_zero_horizon_helpers_are_safe():
    assert _fv_of_sip(10_000, 11.0, 0) == 0.0
    assert _sip_for_target(1_000_000, 11.0, 0) == 0.0


# --------------------------------------------------------------------------- #
# Corpus sizing
# --------------------------------------------------------------------------- #

def test_expenses_are_inflated_to_the_retirement_date():
    out = advise(age=32, monthly_expenses=55_000)
    assert out["monthly_need_at_retirement"] > 55_000 * EXPENSE_REPLACEMENT_RATIO * 3


def test_only_a_share_of_expenses_carries_into_retirement():
    """Commuting and child costs fall away; healthcare rises."""
    assert EXPENSE_REPLACEMENT_RATIO < 1.0


def test_required_corpus_grows_with_a_longer_retirement():
    short = advise(life_expectancy=75)["required_corpus"]
    long = advise(life_expectancy=95)["required_corpus"]
    assert long > short


def test_higher_inflation_needs_a_bigger_corpus():
    low = advise(inflation=4.0)["required_corpus"]
    high = advise(inflation=8.0)["required_corpus"]
    assert high > low


def test_assumptions_are_reported_for_review():
    """A reviewer must be able to disagree with the numbers explicitly."""
    out = advise()
    assert set(out["assumptions"]) >= {
        "inflation", "pre_retirement_return", "post_retirement_return",
        "retirement_age", "life_expectancy",
    }


# --------------------------------------------------------------------------- #
# Gap and contribution
# --------------------------------------------------------------------------- #

def test_a_saver_with_nothing_is_critically_behind():
    out = advise()
    assert out["status"] == "Critically behind"
    assert out["readiness"] == 0
    assert out["additional_monthly_required"] > 0


def test_existing_corpus_reduces_the_required_sip():
    nothing = advise(current_corpus=0)["additional_monthly_required"]
    something = advise(current_corpus=5_000_000)["additional_monthly_required"]
    assert something < nothing


def test_existing_contributions_reduce_the_gap():
    none = advise(current_monthly_contribution=0)["gap"]
    some = advise(current_monthly_contribution=25_000)["gap"]
    assert some < none


def test_a_well_funded_saver_is_on_track():
    out = advise(age=32, current_corpus=30_000_000, current_monthly_contribution=60_000)
    assert out["on_track"] is True
    assert out["status"] == "On track"
    assert out["gap"] == 0
    assert out["additional_monthly_required"] == 0


def test_readiness_is_bounded():
    for corpus in (0, 1_000_000, 500_000_000):
        assert 0 <= advise(current_corpus=corpus)["readiness"] <= 1


def test_the_required_sip_actually_closes_the_gap():
    """Round-trip: investing the recommended amount must reach the target."""
    out = advise(age=32, current_corpus=1_000_000, current_monthly_contribution=10_000)
    projected = (
        _future_value(1_000_000, 11.0, out["years_to_retire"])
        + _fv_of_sip(10_000 + out["additional_monthly_required"], 11.0, out["years_to_retire"])
    )
    assert projected == pytest.approx(out["required_corpus"], rel=0.01)


def test_total_required_is_current_plus_additional():
    out = advise(current_monthly_contribution=10_000)
    assert out["total_monthly_required"] == pytest.approx(
        10_000 + out["additional_monthly_required"], abs=0.01
    )


# --------------------------------------------------------------------------- #
# Drawdown phase
# --------------------------------------------------------------------------- #

def test_a_retiree_gets_a_drawdown_assessment_not_an_accumulation_plan():
    out = advise(age=65, current_corpus=20_000_000)
    assert out["phase"] == "drawdown"
    assert out["years_to_retire"] == 0
    assert "sustainable_annual_withdrawal" in out


def test_a_large_corpus_sustains_the_retirees_needs():
    out = advise(age=65, monthly_expenses=40_000, current_corpus=50_000_000)
    assert out["on_track"] is True
    assert out["shortfall"] == 0


def test_a_thin_corpus_shows_a_drawdown_shortfall():
    out = advise(age=65, monthly_expenses=60_000, current_corpus=500_000)
    assert out["on_track"] is False
    assert out["shortfall"] > 0


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(salaried_profile):
    patch = retirement_node(new_state(salaried_profile))
    assert set(patch) == {"retirement_result"}
    json.dumps(patch)


def test_node_reads_the_retirement_corpus_not_the_emergency_fund():
    """Runway is not a retirement asset; counting it would overstate readiness."""
    runway_only = UserProfile(user_id="r", age=40, essential_expenses=50_000,
                              existing_emergency_fund=2_000_000, retirement_corpus=0)
    invested = runway_only.model_copy(update={"retirement_corpus": 2_000_000})

    assert (
        retirement_node(new_state(invested))["retirement_result"]["projected_corpus"]
        > retirement_node(new_state(runway_only))["retirement_result"]["projected_corpus"]
    )


def test_node_uses_the_profiles_stated_contribution():
    profile = UserProfile(user_id="c", age=35, essential_expenses=40_000,
                          monthly_investment=20_000)
    result = retirement_node(new_state(profile))["retirement_result"]
    assert result["current_monthly_contribution"] == 20_000


def test_node_survives_the_zero_profile(zero_profile):
    result = retirement_node(new_state(zero_profile))["retirement_result"]
    assert result["phase"] in {"accumulation", "drawdown"}
