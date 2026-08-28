"""
Credit Card Intelligence -> Tier 6: Explanation Agent.

Produces the numbered explanation the doc specifies:

    1. Regalia Gold earns Rs 18,400/year from your spending.
    2. You will use 9 of 12 lounge visits.
    3. MMT Black membership adds Rs 2,000 value.
    4. Net expected benefit: Rs 15,900 after annual fee.
    5. Better than Millennia by Rs 4,300.

The explanation is built deterministically from the Tier 2-5 outputs first. The
LLM only rewrites that draft into prose, and only if a provider is available --
so the numbers can never be invented, and an outage degrades to the plain
numbered list rather than to nothing.

Two things this deliberately says out loud:

  * **the realisation gap** -- how much of the card's advertised value this user
    will not collect, because that is the number card marketing omits;
  * **the Risk Agent's objection**, when there is one. A recommendation that
    hides its own strongest counter-argument is not an explanation.
"""

from __future__ import annotations

from typing import Any

from ..common import llm

EXPLAIN_SYSTEM = (
    "You explain a credit card recommendation to an Indian consumer with no "
    "finance background. Rewrite the numbered points into short, plain prose. "
    "Rules: keep EVERY number exactly as given, never add a number, keep the "
    "numbered structure, stay under 140 words, no jargon."
)


def build_explanation(
    ranking: dict[str, Any],
    evaluations: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
    deliberation: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the numbered explanation for the winning card. Pure and deterministic.
    """
    if not ranking.get("ranked"):
        return {
            "status": "nothing to explain",
            "reason": ranking.get("status", "no ranked cards"),
            "points": [],
            "text": "",
        }

    winner = ranking["ranked"][0]
    name = winner["card_name"]
    evaluation = next((e for e in evaluations if e["card_name"] == name), {})
    simulation = next((s for s in simulations if s["card_name"] == name), {})

    rewards = evaluation.get("rewards", {})
    lounge = evaluation.get("lounge", {})
    membership = evaluation.get("membership", {})
    cost = evaluation.get("cost", {})

    points: list[str] = []

    # 1. rewards from actual spending
    points.append(
        f"{name} earns about Rs {rewards.get('annual_rewards', 0):,.0f} a year "
        "from your spending."
    )

    # 2. lounge, offered vs used
    if lounge.get("visits_offered"):
        points.append(
            f"You will use {lounge.get('visits_used', 0)} of "
            f"{lounge['visits_offered']} lounge visits, worth about "
            f"Rs {lounge.get('lounge_value', 0):,.0f}."
        )

    # 3. memberships, at realised value
    if membership.get("membership_value", 0) > 0:
        names = ", ".join(i["benefit"] for i in membership.get("items", [])[:2])
        points.append(
            f"{names or 'Bundled benefits'} add about "
            f"Rs {membership['membership_value']:,.0f} of value you will actually use."
        )

    # 4. the net figure, after cost
    fee_note = (
        "with the annual fee waived by your spending"
        if cost.get("fee_waived") else
        f"after the Rs {cost.get('effective_fee', 0):,.0f} annual fee"
    )
    points.append(
        f"Net expected benefit: Rs {evaluation.get('net_annual_value', 0):,.0f} "
        f"a year {fee_note}."
    )

    # 5. margin over the runner-up
    if ranking.get("runner_up") and ranking.get("margin_over_runner_up") is not None:
        points.append(
            f"Better than {ranking['runner_up']} by about "
            f"Rs {ranking['margin_over_runner_up']:,.0f} a year."
        )

    # 6. the honest range
    if simulation:
        points.append(
            f"In a bad year this falls to about Rs {simulation.get('worst', 0):,.0f}; "
            f"in a good year it reaches Rs {simulation.get('best', 0):,.0f}."
        )

    # 7. what the card advertises but you will not collect
    gap = float(evaluation.get("realisation_gap", 0) or 0)
    if gap > 1_000:
        points.append(
            f"The card advertises about Rs {gap:,.0f} more in benefits than your "
            "usage pattern will realise, mostly unused perks."
        )

    # 8. the strongest argument against
    objections = deliberation.get("objections", []) or []
    objection = next((o for o in objections if o["card_name"] == name), None)
    if objection:
        points.append(
            "One caution: the risk check flags this card as a stretch for your "
            "current budget or usage."
        )

    return {
        "status": "explained",
        "card_name": name,
        "final_score_percent": winner["final_score_percent"],
        "decisive_component": ranking.get("decisive_component"),
        "points": points,
        "text": "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)),
        "has_objection": objection is not None,
    }


def explain(
    explanation: dict[str, Any],
    provider: llm.Provider = "groq",
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Rewrite the numbered draft into prose. Falls back to the draft on failure.
    """
    draft = explanation.get("text", "")
    if not draft:
        return {**explanation, "prose": "", "method": "empty"}
    if not use_llm:
        return {**explanation, "prose": draft, "method": "deterministic"}

    try:
        text = llm.chat(draft, provider=provider, system=EXPLAIN_SYSTEM)
    except Exception as exc:
        return {**explanation, "prose": draft, "method": f"fallback: {exc!r}"}

    return {
        **explanation,
        "prose": (text or "").strip() or draft,
        "method": "llm",
    }
