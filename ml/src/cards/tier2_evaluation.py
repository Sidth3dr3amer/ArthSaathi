"""
Credit Card Intelligence -> Tier 2: Card Evaluation Engine.

    Reward Simulation | Lounge Valuation | Membership Valuation | Cost Agent

Every card enters this pipeline and comes out with a realised value, not a
brochure value. That distinction is the whole point of the tier.

A card advertising 12 lounge visits and a Rs 4,200 hotel membership is quoting
what it *offers*. Tier 2 asks what this specific user will *use*, applying the
Tier 1 twin: someone flying four segments a year realises three lounge visits,
not twelve, and a hotel membership is worth nothing to someone who does not
stay in hotels.

The Reward Simulation agent delegates the reward arithmetic to
`councils.growth.credit_card.calculate_card_value`, which is already tested
against the four curated cards. This tier adds the realisation discounts on top
rather than reimplementing the maths.
"""

from __future__ import annotations

from typing import Any

from ..councils.growth.credit_card import calculate_card_value

#: Rupee value of one domestic lounge visit (typical paid-access rate).
DOMESTIC_LOUNGE_VALUE = 900.0
INTERNATIONAL_LOUNGE_VALUE = 2_400.0

#: Share of a membership's face value a user actually realises, by travel profile.
#: Memberships are the most over-claimed benefit in card marketing.
MEMBERSHIP_REALISATION = {"high": 0.75, "medium": 0.40, "low": 0.15}

#: Share of a movie/dining/golf perk realised. These need local presence and
#: deliberate use, so realisation is low even for engaged users.
LIFESTYLE_REALISATION = 0.35

#: Assumed share of a revolving balance carried on a NEW card in year one.
REVOLVING_MIGRATION = 0.30


def reward_simulation_advisor(
    card: dict[str, Any],
    engine_profile: dict[str, Any],
    spend_analysis: dict[str, Any],
    twin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Expected annual reward value. Pure and deterministic.

    Reuses the tested reward engine, then discounts to zero when the twin says
    the user revolves a balance -- rewards earned while paying 42% interest are
    not a benefit, they are a rounding error on a loss.
    """
    breakdown = calculate_card_value(card, engine_profile, spend_analysis)

    reward_components = {
        key: float(breakdown.get(key, 0) or 0)
        for key in ("base_rewards", "utility_bonus", "travel_bonus",
                    "welcome_bonus", "forex_savings")
        if breakdown.get(key)
    }
    annual_rewards = sum(reward_components.values())

    realisation = 1.0
    note = None
    if twin and not twin.get("rewards_are_real", True):
        realisation = 0.0
        note = (
            "Rewards discounted to zero: the user revolves a balance at a rate "
            "that exceeds any reward return."
        )

    return {
        "annual_rewards": round(annual_rewards * realisation, 2),
        "headline_rewards": round(annual_rewards, 2),
        "components": {k: round(v, 2) for k, v in reward_components.items()},
        "realisation_factor": realisation,
        "note": note,
        "_breakdown": breakdown,
    }


def lounge_valuation_advisor(
    card: dict[str, Any], twin: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Visits actually used x value per visit. Pure and deterministic.

    The doc's formula is `visits x value_per_visit`; the twin caps `visits` at
    what the user's travel pattern supports.
    """
    domestic = int(card.get("domestic_lounge_visits", 0) or 0)
    international = int(card.get("international_lounge_visits", 0) or 0)
    offered = domestic + international

    usable = twin.get("realistic_lounge_visits", offered) if twin else offered
    used = min(offered, max(int(usable), 0))

    # International visits are worth more, so allocate usage to them first.
    used_international = min(used, international)
    used_domestic = used - used_international

    value = (
        used_domestic * DOMESTIC_LOUNGE_VALUE
        + used_international * INTERNATIONAL_LOUNGE_VALUE
    )

    return {
        "lounge_value": round(value, 2),
        "visits_offered": offered,
        "visits_used": used,
        "visits_wasted": max(offered - used, 0),
        "domestic_used": used_domestic,
        "international_used": used_international,
        "priority_pass": bool(card.get("priority_pass_included", False)),
        "utilisation": round(used / offered, 4) if offered else 0.0,
    }


def membership_valuation_advisor(
    card: dict[str, Any], twin: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Value of bundled memberships and lifestyle perks, discounted by likely use.
    Pure and deterministic.
    """
    travel_profile = (twin or {}).get("travel_profile", "low")
    travel_realisation = MEMBERSHIP_REALISATION.get(travel_profile, 0.15)

    items: list[dict[str, Any]] = []

    for key, label, realisation in (
        ("hotel_membership_value", "Hotel membership", travel_realisation),
        ("airline_membership_value", "Airline membership", travel_realisation),
        ("movie_benefit_value", "Movie benefit", LIFESTYLE_REALISATION),
        ("golf_benefit_value", "Golf benefit", LIFESTYLE_REALISATION),
        ("dining_program_value", "Dining programme", LIFESTYLE_REALISATION),
        ("milestone_value_annual", "Milestone benefit", 1.0),
        ("renewal_bonus_value", "Renewal bonus", 1.0),
    ):
        face = float(card.get(key, 0) or 0)
        if face <= 0:
            continue
        items.append({
            "benefit": label,
            "face_value": round(face, 2),
            "realisation": realisation,
            "realised_value": round(face * realisation, 2),
        })

    realised = sum(i["realised_value"] for i in items)
    face_total = sum(i["face_value"] for i in items)

    return {
        "membership_value": round(realised, 2),
        "face_value": round(face_total, 2),
        "value_gap": round(face_total - realised, 2),
        "items": items,
        "travel_profile_applied": travel_profile,
    }


def cost_agent_advisor(
    card: dict[str, Any],
    engine_profile: dict[str, Any],
    twin: dict[str, Any] | None = None,
    reward_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Total annual cost: fee, forex, interest risk, hidden costs.
    Pure and deterministic.

    Interest risk is the line most card comparisons omit and the one that
    dominates for a revolving user, so it is computed explicitly rather than
    folded into a footnote.
    """
    annual_fee = float(card.get("annual_fee", 0) or 0)
    joining_fee = float(card.get("joining_fee", 0) or 0)
    waiver_spend = float(card.get("fee_waiver_spend", 0) or 0)

    annual_spend = float((twin or {}).get("annual_spend", 0) or 0)
    fee_waived = bool(waiver_spend and annual_spend >= waiver_spend)
    effective_fee = 0.0 if fee_waived else annual_fee

    forex_markup = float(card.get("forex_markup", 0) or 0)
    forex_spend = float((twin or {}).get("forex_exposure_annual", 0) or 0)
    forex_cost = forex_spend * forex_markup / 100

    # Interest risk: a share of any existing revolving balance migrates to a new
    # card in year one. Card APRs cluster around 42% in the Indian market.
    interest_risk = 0.0
    if twin and twin.get("revolves_balance"):
        interest_risk = float(twin.get("annual_interest_cost", 0)) * REVOLVING_MIGRATION

    total = effective_fee + joining_fee + forex_cost + interest_risk

    return {
        "cost": round(total, 2),
        "annual_fee": round(annual_fee, 2),
        "joining_fee": round(joining_fee, 2),
        "effective_fee": round(effective_fee, 2),
        "fee_waived": fee_waived,
        "fee_waiver_spend": round(waiver_spend, 2),
        "spend_shortfall_for_waiver": round(
            max(waiver_spend - annual_spend, 0), 2
        ) if waiver_spend else 0.0,
        "forex_cost": round(forex_cost, 2),
        "forex_markup_percent": forex_markup,
        "interest_risk": round(interest_risk, 2),
    }


def evaluate_card(
    card: dict[str, Any],
    engine_profile: dict[str, Any],
    spend_analysis: dict[str, Any],
    twin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all four Tier 2 agents against one card."""
    rewards = reward_simulation_advisor(card, engine_profile, spend_analysis, twin)
    lounge = lounge_valuation_advisor(card, twin)
    membership = membership_valuation_advisor(card, twin)
    cost = cost_agent_advisor(card, engine_profile, twin, rewards.get("_breakdown"))

    gross = rewards["annual_rewards"] + lounge["lounge_value"] + membership["membership_value"]
    net = gross - cost["cost"]

    # Brochure value: every offered lounge visit and every membership at face.
    # Reported alongside the realised figure so the gap between what a card
    # advertises and what this user will actually collect is explicit.
    headline_lounge = (
        lounge["domestic_used"] + lounge["visits_wasted"]
    ) * DOMESTIC_LOUNGE_VALUE + lounge["international_used"] * INTERNATIONAL_LOUNGE_VALUE
    headline_gross = (
        rewards["headline_rewards"] + headline_lounge + membership["face_value"]
    )

    return {
        "card_name": card.get("card_name"),
        "card_type": card.get("card_type"),
        "card_tier": card.get("card_tier"),
        "rewards": {k: v for k, v in rewards.items() if k != "_breakdown"},
        "lounge": lounge,
        "membership": membership,
        "cost": cost,
        "gross_value": round(gross, 2),
        "net_annual_value": round(net, 2),
        "headline_gross": round(headline_gross, 2),
        "realisation_gap": round(headline_gross - gross, 2),
    }


def evaluate_all(
    cards: list[dict[str, Any]],
    engine_profile: dict[str, Any],
    spend_analysis: dict[str, Any],
    twin: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every card, best net value first."""
    evaluated = [evaluate_card(c, engine_profile, spend_analysis, twin) for c in cards]
    evaluated.sort(key=lambda e: e["net_annual_value"], reverse=True)
    return evaluated
