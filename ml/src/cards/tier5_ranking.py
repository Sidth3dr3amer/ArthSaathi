"""
Credit Card Intelligence -> Tier 5: Ranking Engine.

    FinalScore = 0.35 x NetAnnualValue
               + 0.20 x UserMatch
               + 0.15 x ApprovalProbability
               + 0.15 x FutureValue
               + 0.15 x AgentConsensus

The five weights come straight from the design doc. Everything interesting here
is in how the five components get onto a common 0..1 scale, because a weighted
sum of quantities in different units is meaningless.

  NetAnnualValue      normalised against the best card in THIS comparison, so
                      the score answers "how close to the best available option"
                      rather than being anchored to an arbitrary rupee ceiling
  UserMatch           card's stated use case against the user's dominant spend
  ApprovalProbability income and eligibility headroom against the card's bar
  FutureValue         Tier 3's downside, penalised for volatility -- a card whose
                      value collapses in a bad year scores lower than its average
                      suggests
  AgentConsensus      Tier 4's weighted panel agreement

Pure and deterministic throughout.
"""

from __future__ import annotations

from typing import Any

#: The doc's weights. Kept as data so a reviewer can see them and disagree.
WEIGHTS: dict[str, float] = {
    "net_annual_value": 0.35,
    "user_match": 0.20,
    "approval_probability": 0.15,
    "future_value": 0.15,
    "agent_consensus": 0.15,
}

#: Card type -> the spend buckets it is designed for.
TYPE_AFFINITY: dict[str, set[str]] = {
    "CASHBACK": {"online", "offline", "grocery", "utility", "dining"},
    "TRAVEL": {"travel", "international"},
    "REWARDS": {"online", "dining", "offline", "travel"},
    "FUEL": {"fuel"},
    "SHOPPING": {"online", "offline"},
    "LIFESTYLE": {"dining", "online"},
}


def _normalise(value: float, best: float, worst: float) -> float:
    """Scale a value to 0..1 against the range present in this comparison."""
    if best <= worst:
        return 1.0 if value >= best else 0.0
    return max(0.0, min((value - worst) / (best - worst), 1.0))


def user_match_score(
    card: dict[str, Any], evaluation: dict[str, Any], spending: dict[str, Any]
) -> dict[str, Any]:
    """How well the card's design matches where this user actually spends."""
    card_type = (evaluation.get("card_type") or "").upper()
    affinity = TYPE_AFFINITY.get(card_type, set())

    buckets = spending.get("buckets_annual", {}) or {}
    total = sum(buckets.values())
    matched = sum(v for k, v in buckets.items() if k in affinity)
    share = matched / total if total else 0.0

    dominant = spending.get("dominant_category")
    dominant_hit = bool(dominant and dominant in affinity)

    score = share * 0.7 + (0.3 if dominant_hit else 0.0)
    return {
        "score": round(min(score, 1.0), 4),
        "card_type": card_type,
        "matched_spend_share": round(share, 4),
        "dominant_category": dominant,
        "dominant_matched": dominant_hit,
    }


def approval_probability_score(
    card: dict[str, Any], profiler: dict[str, Any], twin: dict[str, Any]
) -> dict[str, Any]:
    """Likelihood the bank actually issues this card to this user."""
    required = float(card.get("income_requirement", 0) or 0)
    income = float(profiler.get("income", 0) or 0)

    if card.get("invite_only"):
        return {"score": 0.15, "reason": "invite-only card", "income_ratio": None}

    if required <= 0:
        ratio = 2.0                    # no stated bar
    elif income <= 0:
        return {"score": 0.2, "reason": "income unknown", "income_ratio": None}
    else:
        ratio = income / required

    if ratio >= 2.0:
        score = 0.95
    elif ratio >= 1.5:
        score = 0.85
    elif ratio >= 1.0:
        score = 0.70
    elif ratio >= 0.8:
        score = 0.35
    else:
        score = 0.10

    reasons = []
    if not twin.get("approval_headroom", True):
        score *= 0.7
        reasons.append("high existing credit utilisation")
    if not profiler.get("salaried", True) and not card.get("self_employed_eligible", True):
        score *= 0.5
        reasons.append("card prefers salaried applicants")

    return {
        "score": round(max(0.0, min(score, 1.0)), 4),
        "income_ratio": round(ratio, 3),
        "income_requirement": required,
        "reason": "; ".join(reasons) or "meets stated criteria",
    }


def future_value_score(
    simulation: dict[str, Any], best_worst: float, worst_worst: float
) -> dict[str, Any]:
    """
    Durability of the card's value. Built on Tier 3's WORST case, not its
    average, and penalised for volatility -- a card that only pays off in a good
    year is a worse recommendation than its mean suggests.
    """
    downside = float(simulation.get("worst", 0))
    volatility = float(simulation.get("volatility", 0))
    base = _normalise(downside, best_worst, worst_worst)
    score = base * (1 - volatility * 0.3)

    return {
        "score": round(max(0.0, min(score, 1.0)), 4),
        "worst_case": round(downside, 2),
        "volatility": volatility,
        "volatility_penalty": round(base - score, 4),
        "downside_is_negative": bool(simulation.get("downside_is_negative", False)),
    }


def rank_cards(
    evaluations: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
    deliberation: dict[str, Any],
    cards: list[dict[str, Any]],
    profiler: dict[str, Any],
    spending: dict[str, Any],
    twin: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Apply the doc's weighted formula. Pure and deterministic.

    Returns cards ranked by FinalScore, each carrying its full component
    breakdown so Tier 6 can explain the ranking rather than assert it.
    """
    if not evaluations:
        return {"status": "no cards to rank", "ranked": [], "weights": dict(WEIGHTS)}

    by_name_card = {c.get("card_name"): c for c in cards}
    by_name_sim = {s["card_name"]: s for s in simulations}
    consensus = deliberation.get("consensus", {}) or {}

    nets = [float(e["net_annual_value"]) for e in evaluations]
    best_net, worst_net = max(nets), min(nets)

    worsts = [float(s.get("worst", 0)) for s in simulations] or [0.0]
    best_worst, worst_worst = max(worsts), min(worsts)

    ranked: list[dict[str, Any]] = []
    for evaluation in evaluations:
        name = evaluation["card_name"]
        card = by_name_card.get(name, {})
        simulation = by_name_sim.get(name, {})

        net_component = _normalise(
            float(evaluation["net_annual_value"]), best_net, worst_net
        )
        match = user_match_score(card, evaluation, spending)
        approval = approval_probability_score(card, profiler, twin)
        future = future_value_score(simulation, best_worst, worst_worst)
        agent = float(consensus.get(name, 0.0))

        components = {
            "net_annual_value": round(net_component, 4),
            "user_match": match["score"],
            "approval_probability": approval["score"],
            "future_value": future["score"],
            "agent_consensus": round(agent, 4),
        }
        final = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)

        ranked.append({
            "card_name": name,
            "final_score": round(final, 4),
            "final_score_percent": round(final * 100, 2),
            "components": components,
            "weighted_contributions": {
                k: round(components[k] * WEIGHTS[k], 4) for k in WEIGHTS
            },
            "net_annual_value": evaluation["net_annual_value"],
            "detail": {
                "user_match": match,
                "approval": approval,
                "future_value": future,
                "simulation": {
                    "best": simulation.get("best"),
                    "avg": simulation.get("avg"),
                    "worst": simulation.get("worst"),
                },
            },
        })

    ranked.sort(key=lambda r: r["final_score"], reverse=True)

    top = ranked[:top_n]
    margin = (
        round(ranked[0]["net_annual_value"] - ranked[1]["net_annual_value"], 2)
        if len(ranked) > 1 else None
    )

    return {
        "status": "ranked",
        "weights": dict(WEIGHTS),
        "ranked": top,
        "all_ranked": ranked,
        "winner": ranked[0]["card_name"],
        "margin_over_runner_up": margin,
        "runner_up": ranked[1]["card_name"] if len(ranked) > 1 else None,
        "decisive_component": max(
            ranked[0]["weighted_contributions"],
            key=lambda k: ranked[0]["weighted_contributions"][k],
        ),
    }
