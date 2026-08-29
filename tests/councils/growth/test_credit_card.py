"""
Growth Council -> Credit Card Agent (Tier 2: Card Evaluation Engine).

Golden net-annual-value figures are pinned against the four curated cards in
`CreditCardDataMaker_Final/final_decision/`, so a change to the scoring weights
cannot silently alter a recommendation.
"""

from __future__ import annotations

import json

import pytest

from ml.src.councils.growth.credit_card import (
    analyze_spend_profile,
    calculate_card_value,
    credit_card_node,
    filter_eligible_cards,
    load_card_database,
    profile_to_engine_dict,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture
def card_user() -> UserProfile:
    """Mirrors the notebook's USER_PROFILE literal."""
    return UserProfile(
        user_id="card-user", name="Rahul", age=28, job_type="salaried",
        monthly_income=80_000, max_annual_fee=2_000, prefer_cashback=True,
        travel_frequency="occasional",
        monthly_spend={
            "online_shopping": 8_000, "groceries": 5_000, "dining": 4_000,
            "fuel": 3_000, "utility_bills": 3_000, "travel": 5_000,
        },
        lifestyle_flags={"is_airtel_user": True},
    )


@pytest.fixture
def db():
    """
    The hand-curated four only. The golden net-value figures below are pinned
    against these, so promoting more cards into `card_pool/` must not move them.
    """
    return load_card_database(include_pool=False)


@pytest.fixture
def full_db():
    """Curated cards plus everything promoted out of the raw extraction."""
    return load_card_database()


# --------------------------------------------------------------------------- #
# Card database
# --------------------------------------------------------------------------- #

def test_curated_database_loads_the_four_hand_checked_cards(db):
    assert len(db) == 4
    assert {c["card_name"] for c in db} == {
        "Airtel Axis Bank Credit Card",
        "Axis Bank Ace Credit Card",
        "Axis Bank Atlas Credit Card",
        "Axis Bank Aura Credit Card",
    }


def test_missing_database_directory_returns_empty_not_error(tmp_path):
    assert load_card_database(tmp_path / "nope") == []


def test_malformed_json_is_skipped_not_fatal(tmp_path):
    (tmp_path / "good.json").write_text('{"card_name": "Good"}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    loaded = load_card_database(tmp_path)
    assert [c["card_name"] for c in loaded] == ["Good"]


# --------------------------------------------------------------------------- #
# Profile adaptation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "job_type,expected",
    [("salaried", "salaried"), ("govt", "salaried"),
     ("business", "self_employed"), ("freelancer", "self_employed"),
     ("student", "student")],
)
def test_job_type_maps_to_engine_employment_type(job_type, expected):
    profile = UserProfile(job_type=job_type)
    assert profile_to_engine_dict(profile)["employment_type"] == expected


def test_lifestyle_flags_are_flattened_into_the_engine_dict(card_user):
    assert profile_to_engine_dict(card_user)["is_airtel_user"] is True


# --------------------------------------------------------------------------- #
# Spend analysis
# --------------------------------------------------------------------------- #

def test_spend_analysis_annualises_and_ranks(card_user):
    out = analyze_spend_profile(profile_to_engine_dict(card_user))
    assert out["total_monthly"] == 28_000
    assert out["total_annual"] == 336_000
    assert out["dominant_type"] == "CASHBACK"
    assert out["ranked_types"][0][0] == "CASHBACK"


def test_spend_analysis_survives_an_empty_spend_map():
    """
    The engine guards its ratio maths with `sum(spend.values()) or 1`, so a user
    with no recorded spend reports a sentinel total of 1 rather than dividing by
    zero. Pinned here so the guard is not "tidied away" later.
    """
    out = analyze_spend_profile(profile_to_engine_dict(UserProfile()))
    assert out["total_monthly"] == 1
    assert out["total_annual"] == 12
    assert all(0 <= r <= 1 for r in out["ratios"].values())


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #

def test_eligibility_splits_the_database(card_user, db):
    eligible, rejected = filter_eligible_cards(profile_to_engine_dict(card_user), db)
    assert len(eligible) + len(rejected) == len(db)
    assert len(eligible) == 3


def test_low_income_applicant_is_rejected_from_more_cards(db):
    poor = profile_to_engine_dict(UserProfile(age=28, job_type="salaried", monthly_income=8_000))
    rich = profile_to_engine_dict(UserProfile(age=28, job_type="salaried", monthly_income=500_000))
    assert len(filter_eligible_cards(poor, db)[0]) <= len(filter_eligible_cards(rich, db)[0])


def test_underage_applicant_is_rejected(db):
    minor = profile_to_engine_dict(UserProfile(age=16, monthly_income=80_000))
    eligible, rejected = filter_eligible_cards(minor, db)
    assert rejected


def test_rejections_carry_a_reason(card_user, db):
    _, rejected = filter_eligible_cards(profile_to_engine_dict(card_user), db)
    assert rejected and all(len(r) >= 2 for r in rejected)


# --------------------------------------------------------------------------- #
# Valuation — golden figures
# --------------------------------------------------------------------------- #

def test_valuation_golden_net_values(card_user, db):
    engine_profile = profile_to_engine_dict(card_user)
    analysis = analyze_spend_profile(engine_profile)
    eligible, _ = filter_eligible_cards(engine_profile, db)

    net = {
        c["card_name"]: calculate_card_value(c, engine_profile, analysis)["net_value"]
        for c in eligible
    }
    assert net["Axis Bank Aura Credit Card"] == pytest.approx(4_291.0, abs=1.0)
    assert net["Axis Bank Ace Credit Card"] == pytest.approx(3_780.0, abs=1.0)
    assert net["Airtel Axis Bank Credit Card"] == pytest.approx(2_520.0, abs=1.0)


def test_net_value_is_gross_rewards_minus_fee(card_user, db):
    engine_profile = profile_to_engine_dict(card_user)
    analysis = analyze_spend_profile(engine_profile)
    card = next(c for c in db if c["card_name"] == "Axis Bank Ace Credit Card")
    val = calculate_card_value(card, engine_profile, analysis)
    assert "net_value" in val
    assert val["net_value"] <= val.get("total_value", val["net_value"])


def test_higher_spend_never_reduces_gross_reward_value(db):
    card = next(c for c in db if c["card_name"] == "Axis Bank Ace Credit Card")

    def net_for(multiplier):
        p = profile_to_engine_dict(UserProfile(
            age=30, job_type="salaried", monthly_income=200_000,
            monthly_spend={"groceries": 10_000 * multiplier, "utility_bills": 5_000 * multiplier},
        ))
        return calculate_card_value(card, p, analyze_spend_profile(p))["net_value"]

    assert net_for(4) >= net_for(1)


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_ranks_by_net_value_descending(card_user):
    """Ordering must hold over the full pool, not just the curated four."""
    result = credit_card_node(new_state(card_user, query="which card?"))["credit_card_result"]
    values = [r["net_value"] for r in result["recommendations"]]
    assert values == sorted(values, reverse=True)


def test_the_curated_winner_is_unchanged_by_promotion(card_user, db):
    """
    Pinned against the hand-checked four. Promoting cards into `card_pool/`
    adds candidates; it must never change how an existing card is scored.
    """
    result = credit_card_node(new_state(card_user), cards=db)["credit_card_result"]
    assert result["recommendations"][0]["card"]["card_name"] == "Axis Bank Aura Credit Card"


def test_the_promoted_pool_widens_the_candidate_set(card_user):
    curated = credit_card_node(new_state(card_user), cards=load_card_database(include_pool=False))
    full = credit_card_node(new_state(card_user))
    assert (full["credit_card_result"]["cards_considered"]
            > curated["credit_card_result"]["cards_considered"])
    assert (full["credit_card_result"]["eligible_count"]
            >= curated["credit_card_result"]["eligible_count"])


def test_node_contract_and_counts(card_user, db):
    patch = credit_card_node(new_state(card_user), cards=db)
    assert set(patch) == {"credit_card_result"}
    result = patch["credit_card_result"]
    assert result["cards_considered"] == 4
    assert result["eligible_count"] == 3
    assert result["rejected_count"] == 1
    assert set(result["routing"]) >= {"groceries", "dining", "fuel"}


def test_node_respects_top_n(card_user):
    result = credit_card_node(new_state(card_user), top_n=1)["credit_card_result"]
    assert len(result["recommendations"]) == 1


def test_node_handles_an_empty_database(card_user):
    result = credit_card_node(new_state(card_user), cards=[])["credit_card_result"]
    assert result["cards_considered"] == 0
    assert result["recommendations"] == []
    assert result["routing"] == {}


def test_node_handles_a_profile_with_no_spend(db):
    blank = UserProfile(user_id="blank", age=30, job_type="salaried", monthly_income=60_000)
    result = credit_card_node(new_state(blank), cards=db)["credit_card_result"]
    # sentinel total (see test_spend_analysis_survives_an_empty_spend_map)
    assert result["spend_analysis"]["total_annual"] == 12
    # `max_annual_fee` defaults to 0, so every fee-charging card is filtered out.
    # The agent must return an empty ranking rather than raising.
    assert result["eligible_count"] == 0
    assert result["recommendations"] == []
    assert result["routing"] == {}


def test_raising_the_fee_ceiling_admits_cards(db):
    """Confirms the previous test's empty result is the fee gate, not a bug."""
    willing = UserProfile(user_id="willing", age=30, job_type="salaried",
                          monthly_income=60_000, max_annual_fee=5_000)
    result = credit_card_node(new_state(willing), cards=db)["credit_card_result"]
    assert result["eligible_count"] > 0


def test_node_output_is_json_serialisable(card_user):
    patch = credit_card_node(new_state(card_user))
    json.dumps(patch)


def test_node_does_not_mutate_the_injected_database(card_user, db):
    before = json.dumps(db, sort_keys=True)
    credit_card_node(new_state(card_user), cards=db)
    assert json.dumps(db, sort_keys=True) == before


# --------------------------------------------------------------------------- #
# Cards the user already holds
# --------------------------------------------------------------------------- #

from ml.src.councils.growth.credit_card import (  # noqa: E402
    REVOLVING_RATE_THRESHOLD,
    existing_card_burden,
)


def _revolver(**over) -> UserProfile:
    base = dict(
        user_id="rahul", age=32, job_type="freelancer",
        monthly_income=18_000, essential_expenses=14_000,
        monthly_spend={"dining": 2_000, "fuel": 1_500, "online": 3_000},
        debts=[{
            "name": "Credit card", "debt_type": "credit_card",
            "outstanding_amount": 48_000, "interest_rate": 42.0,
            "minimum_due": 2_400,
        }],
    )
    base.update(over)
    return UserProfile(**base)


def test_a_user_with_no_debts_holds_no_cards():
    burden = existing_card_burden(UserProfile(user_id="u", age=30, monthly_income=50_000))
    assert burden["verdict"] == "no_cards"
    assert burden["revolving"] is False


def test_a_non_card_debt_is_not_counted_as_a_card():
    """A gold loan at 12% is a debt, but it is not a credit card."""
    profile = _revolver(debts=[{
        "name": "Gold loan", "debt_type": "gold_loan",
        "outstanding_amount": 80_000, "interest_rate": 12.0,
    }])
    assert existing_card_burden(profile)["verdict"] == "no_cards"


def test_a_paid_off_card_is_held_but_not_revolving():
    profile = _revolver(debts=[{
        "name": "Old card", "debt_type": "credit_card",
        "outstanding_amount": 0, "interest_rate": 42.0,
    }])
    burden = existing_card_burden(profile)
    assert burden["holds_cards"] is True
    assert burden["verdict"] == "holds_cards_paid_off"


def test_interest_cost_is_computed_from_the_actual_rate():
    burden = existing_card_burden(_revolver())
    assert burden["annual_interest_cost"] == pytest.approx(48_000 * 0.42)
    assert burden["effective_rate"] == pytest.approx(42.0)


def test_a_low_rate_card_balance_is_not_treated_as_revolving():
    profile = _revolver(debts=[{
        "name": "Converted to EMI", "debt_type": "credit_card",
        "outstanding_amount": 48_000, "interest_rate": 14.0,
    }])
    assert 14.0 < REVOLVING_RATE_THRESHOLD
    assert existing_card_burden(profile)["revolving"] is False


def test_a_revolving_user_is_not_told_to_take_a_new_card():
    """
    The headline failure this guards: recommending a second card to someone
    paying 42% on the first. The rewards are worth a fraction of the interest.
    """
    result = credit_card_node(new_state(_revolver()))["credit_card_result"]
    assert result["recommend_new_card"] is False
    assert result["caution"]
    assert "48,000" in result["summary"]
    assert result["summary"] == result["caution"]


def test_the_caution_compares_rewards_against_the_interest_bill(db):
    result = credit_card_node(new_state(_revolver()), cards=db)["credit_card_result"]
    if result["recommendations"]:
        assert "smaller" in result["caution"]
    assert "Clearing that balance" in result["caution"]


def test_a_card_the_user_already_holds_is_not_recommended_back(db):
    held = db[0]["card_name"]
    profile = _revolver(
        monthly_income=90_000,
        debts=[{
            "name": held, "debt_type": "credit_card",
            "outstanding_amount": 0, "interest_rate": 42.0,
        }],
    )
    result = credit_card_node(new_state(profile), cards=db)["credit_card_result"]
    assert held not in [r["card"]["card_name"] for r in result["recommendations"]]
    assert result["cards_considered"] == len(db) - 1


def test_a_clean_user_still_gets_recommendations():
    """The guard must not suppress advice for someone who carries no balance."""
    profile = _revolver(monthly_income=90_000, essential_expenses=30_000, debts=[])
    result = credit_card_node(new_state(profile))["credit_card_result"]
    assert result["recommend_new_card"] is True
    assert result["caution"] is None
