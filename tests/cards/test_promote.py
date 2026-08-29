"""Promotion of raw extracted card profiles into the engine's scoring shape."""

from __future__ import annotations

from ml.src.cards.promote import (
    MAX_BASE_RATE,
    MIN_BASE_RATE,
    MIN_FEE_WAIVER,
    _base_rate,
    _number,
    promote_card,
)


def _raw(**over):
    base = {
        "card_name": "Test Card",
        "card_type": "Credit Card",
        "annual_fee": "Rs 500",
        "reward_structure": ["1.5% cashback on all other spends", "5% on dining"],
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Fee coercion
# --------------------------------------------------------------------------- #

def test_prose_fees_are_read():
    assert _number("Rs 12,500 plus taxes") == 12_500.0


def test_lifetime_free_is_zero_not_missing():
    assert _number("Lifetime Free") == 0.0


def test_unparseable_fee_is_none_not_zero():
    """Zero and unknown are different; conflating them invents a free card."""
    assert _number("varies by variant") is None


# --------------------------------------------------------------------------- #
# Base rate — the number that anchors every valuation
# --------------------------------------------------------------------------- #

def test_all_other_spends_wins_over_a_category_headline():
    rate = _base_rate(["10% on dining", "1% on all other spends"])
    assert rate == 1.0


def test_category_only_rates_never_become_the_base():
    """
    A card whose profile states only category rates has no derivable base rate,
    and inferring one from "5% on fuel" is how twelve cards came to claim 5% on
    all spend -- outranking every curated card on a rate none of them pay.
    """
    assert _base_rate(["5% on fuel", "2% on travel"]) is None


def test_a_merchant_rate_never_becomes_the_base():
    """The categories are brand names, so no keyword list can catch this."""
    assert _base_rate(["10% cashback on Samsung purchases"]) is None


def test_the_lowest_stated_base_rate_wins_when_sources_disagree():
    """
    HDFC Pixel states both "1% on all transactions" and a marketing line
    reading "5% cashback on all spends". A base rate is a floor by definition.
    """
    assert _base_rate(["5% cashback on all spends",
                       "1% CashBack on all transactions"]) == 1.0


def test_a_milestone_bonus_is_not_a_base_rate():
    assert _base_rate(["15% milestone bonus", "1% on all other spends"]) == 1.0


def test_implausible_rates_are_rejected_not_clamped():
    """
    The first version promoted Axis Magnus at a 15% base earn rate, read out of
    a milestone line. A wrong rate does not fail loudly -- it silently makes
    that card dominate every ranking it appears in.
    """
    assert _base_rate(["100% fee waiver on all spends"]) is None


def test_a_card_with_only_an_implausible_rate_is_not_promoted():
    card = promote_card(
        _raw(reward_structure=["Earn 15X rewards", "100% welcome bonus"]), "axis"
    )
    assert card is None


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #

def test_promoted_card_carries_a_scoreable_base_rate():
    card = promote_card(_raw(), "hdfc")
    assert MIN_BASE_RATE <= card["base_reward_rate"] <= MAX_BASE_RATE


def test_category_rates_are_kept_only_when_above_base():
    card = promote_card(_raw(), "hdfc")
    assert card.get("dining_reward_rate", 0) > card["base_reward_rate"]


def test_unknown_card_type_falls_back_to_the_neutral_default():
    """'Credit Card' carries no signal for the scorer, so it is not preserved."""
    assert promote_card(_raw(), "hdfc")["card_type"] == "REWARDS"


def test_a_real_card_type_survives():
    assert promote_card(_raw(card_type="Cashback"), "hdfc")["card_type"] == "CASHBACK"


def test_eligibility_is_flagged_unconfirmed_never_invented():
    """
    The raw `eligibility` field contains card NETWORKS, not employment rules.
    Promoted cards stay permissive and say so, mirroring how the Benefits
    Council treats a rule it cannot evaluate.
    """
    card = promote_card(_raw(), "hdfc")
    assert card["eligibility_confirmed"] is False
    assert card["source"] == "promoted_from_card_attributes"


def test_a_nonsense_fee_waiver_threshold_is_dropped():
    """A waiver at Rs 25 is a misread of 'Rs 25 lakh', not a generous offer."""
    card = promote_card(_raw(fee_waiver_condition="spend 25 lakh"), "axis")
    assert card.get("fee_waiver_spend", MIN_FEE_WAIVER) >= MIN_FEE_WAIVER


def test_a_card_with_no_reward_information_is_skipped():
    assert promote_card(_raw(reward_structure=[]), "hdfc") is None


def test_a_nameless_card_is_skipped():
    assert promote_card(_raw(card_name=None), "hdfc") is None
