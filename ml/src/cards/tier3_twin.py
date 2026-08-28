"""
Credit Card Intelligence -> Tier 3: Financial Twin Simulation.

    Current Behaviour -> 12-Month Simulation -> Best / Average / Worst

The doc calls this the differentiator, and the reason is that a single expected
value hides the thing users actually care about: how much the answer moves if
the year does not go to plan.

A card whose value swings from Rs 22,000 to Rs 9,500 depending on whether you
travel is a different proposition from one that reliably returns Rs 15,000, even
when their averages match. Tier 5 uses that spread -- a card is penalised for
depending on behaviour the user may not sustain.

Three drivers vary across scenarios:

  spend        a good year spends more, earning more rewards
  lounge use   travel plans change; the worst case is the visits going unused
  fee waiver   a spend-linked waiver is earned in a good year and missed in a bad one

Seeded and deterministic. No Monte Carlo here -- three named scenarios are more
explicable to a user than a distribution, and Tier 5 only needs the spread.
"""

from __future__ import annotations

from typing import Any

from .tier2_evaluation import (
    DOMESTIC_LOUNGE_VALUE,
    INTERNATIONAL_LOUNGE_VALUE,
)

#: Scenario -> (spend multiplier, share of usable lounge visits taken,
#:              share of realised memberships used)
SCENARIOS: dict[str, tuple[float, float, float]] = {
    "best": (1.20, 1.00, 1.00),
    "average": (1.00, 0.75, 0.80),
    "worst": (0.80, 0.25, 0.30),
}


def simulate_card(
    evaluation: dict[str, Any],
    twin: dict[str, Any] | None = None,
    months: int = 12,
) -> dict[str, Any]:
    """
    Project one card's net value across three scenarios. Pure and deterministic.

    `evaluation` is a Tier 2 result. Rewards scale with spend; lounge and
    membership value scale with usage; the fee waiver flips when a
    spend-multiplied year crosses (or misses) the threshold.
    """
    rewards = float(evaluation["rewards"]["annual_rewards"])
    lounge = evaluation["lounge"]
    membership_value = float(evaluation["membership"]["membership_value"])
    cost = evaluation["cost"]

    base_spend = float((twin or {}).get("annual_spend", 0) or 0)
    waiver_spend = float(cost.get("fee_waiver_spend", 0) or 0)
    annual_fee = float(cost.get("annual_fee", 0) or 0)
    fixed_costs = (
        float(cost.get("joining_fee", 0) or 0)
        + float(cost.get("forex_cost", 0) or 0)
        + float(cost.get("interest_risk", 0) or 0)
    )

    scale = months / 12

    results: dict[str, dict[str, Any]] = {}
    for name, (spend_mult, lounge_use, membership_use) in SCENARIOS.items():
        scenario_rewards = rewards * spend_mult * scale

        visits = int(round(lounge["visits_used"] * lounge_use))
        used_international = min(visits, lounge["international_used"])
        used_domestic = visits - used_international
        scenario_lounge = (
            used_domestic * DOMESTIC_LOUNGE_VALUE
            + used_international * INTERNATIONAL_LOUNGE_VALUE
        ) * scale

        scenario_membership = membership_value * membership_use * scale

        scenario_spend = base_spend * spend_mult
        waived = bool(waiver_spend and scenario_spend >= waiver_spend)
        scenario_fee = 0.0 if waived else annual_fee

        gross = scenario_rewards + scenario_lounge + scenario_membership
        net = gross - (scenario_fee * scale) - (fixed_costs * scale)

        results[name] = {
            "gross": round(gross, 2),
            "net": round(net, 2),
            "rewards": round(scenario_rewards, 2),
            "lounge": round(scenario_lounge, 2),
            "membership": round(scenario_membership, 2),
            "fee_paid": round(scenario_fee * scale, 2),
            "fee_waived": waived,
            "annual_spend": round(scenario_spend, 2),
            "lounge_visits": visits,
        }

    best = results["best"]["net"]
    average = results["average"]["net"]
    worst = results["worst"]["net"]
    spread = best - worst

    return {
        "card_name": evaluation.get("card_name"),
        "months": months,
        "best": round(best, 2),
        "avg": round(average, 2),
        "worst": round(worst, 2),
        "spread": round(spread, 2),
        # 0..1. High volatility means the card's value depends on behaviour the
        # user may not sustain, which Tier 5 penalises.
        "volatility": round(
            min(spread / abs(average), 2.0) / 2.0, 4
        ) if average else 1.0,
        "downside_is_negative": worst < 0,
        "scenarios": results,
    }


def simulate_all(
    evaluations: list[dict[str, Any]],
    twin: dict[str, Any] | None = None,
    months: int = 12,
) -> list[dict[str, Any]]:
    """Simulate every evaluated card."""
    return [simulate_card(e, twin, months) for e in evaluations]
