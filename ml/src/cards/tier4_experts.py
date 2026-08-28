"""
Credit Card Intelligence -> Tier 4: Deliberation Layer.

    Cashback Expert | Travel Expert | Premium Expert | Cost Optimizer | Risk Agent

Five experts each advocate for a different card, and their disagreement is the
signal. A card every expert ranks highly is a safe recommendation; a card only
the Premium Expert likes is a card whose case depends entirely on one dimension.
Tier 5 consumes that agreement as `AgentConsensus`.

Each expert has a **deterministic scoring function** -- how well a card fits that
expert's thesis -- so the debate happens without an LLM and is reproducible. The
LLM is used only to phrase an argument, and only if a provider is available.
That keeps the ranking testable and the reasoning explainable.

The Risk Agent is the dissenting voice by design. It does not advocate for a
card; it argues against the ones whose case does not survive the Tier 1 twin --
the doc's own example being "user unlikely to justify annual fee".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..common import llm

#: Weight of an expert's vote in the consensus. The Risk Agent counts double
#: because a veto is more informative than an endorsement -- four experts liking
#: a card the user cannot afford should not outvote the one that noticed.
EXPERT_WEIGHT = {
    "cashback": 1.0,
    "travel": 1.0,
    "premium": 1.0,
    "cost_optimizer": 1.0,
    "risk": 2.0,
}


@dataclass(frozen=True)
class Expert:
    """One voice in the Tier 4 debate."""

    key: str
    label: str
    thesis: str
    score: Callable[[dict[str, Any], dict[str, Any]], float]


#: Reward components that recur every year regardless of behaviour. A welcome
#: bonus is one-time and a milestone benefit is conditional on hitting a spend
#: threshold, so neither belongs in a "direct, unconditional earning" thesis.
RECURRING_REWARD_COMPONENTS = ("base_rewards", "utility_bonus")


def _cashback_score(ev: dict[str, Any], twin: dict[str, Any]) -> float:
    """
    Favours direct, unconditional reward earning.

    Counts only recurring components: including a one-time welcome bonus or a
    conditional milestone benefit would let a premium travel card win this
    expert's vote, which contradicts the thesis it is supposed to argue.
    """
    components = ev["rewards"].get("components", {})
    recurring = sum(
        float(components.get(k, 0) or 0) for k in RECURRING_REWARD_COMPONENTS
    )
    spend = max(float(twin.get("annual_spend", 0) or 0), 1.0)
    effective_rate = recurring / spend
    bonus = 0.25 if ev.get("card_type", "").upper() == "CASHBACK" else 0.0
    return min(effective_rate * 12 + bonus, 1.0)


def _travel_score(ev: dict[str, Any], twin: dict[str, Any]) -> float:
    """Favours lounge access and forex terms, weighted by actual travel."""
    if twin.get("travel_profile") == "low":
        return min(float(ev["lounge"]["lounge_value"]) / 20_000, 0.35)
    lounge = float(ev["lounge"]["lounge_value"]) / 15_000
    forex = max(0.0, (3.5 - float(ev["cost"]["forex_markup_percent"])) / 3.5) * 0.4
    return min(lounge + forex, 1.0)


def _premium_score(ev: dict[str, Any], twin: dict[str, Any]) -> float:
    """Favours total realised benefit, but only if the fee is affordable."""
    if not twin.get("can_absorb_premium_fee", False):
        return 0.1
    gross = float(ev["gross_value"])
    return min(gross / 40_000, 1.0)


def _cost_optimizer_score(ev: dict[str, Any], twin: dict[str, Any]) -> float:
    """Favours the best value per rupee of cost."""
    cost = float(ev["cost"]["cost"])
    net = float(ev["net_annual_value"])
    if cost <= 0:
        return 1.0 if net > 0 else 0.5
    return min(max(net / cost, 0.0) / 10, 1.0)


def _risk_score(ev: dict[str, Any], twin: dict[str, Any]) -> float:
    """
    Confidence that this card is SAFE for this user. Low means the Risk Agent
    objects.
    """
    score = 1.0
    fee = float(ev["cost"]["effective_fee"])
    tolerance = float(twin.get("fee_tolerance", 0) or 0)

    if fee > 0 and tolerance <= 0:
        score -= 0.5
    elif fee > tolerance > 0:
        score -= 0.3

    if not twin.get("rewards_are_real", True):
        score -= 0.4
    if float(ev["net_annual_value"]) <= 0:
        score -= 0.3
    if not twin.get("approval_headroom", True):
        score -= 0.15
    if float(ev["lounge"]["utilisation"]) < 0.3 and ev["lounge"]["visits_offered"] > 6:
        score -= 0.1                   # paying for lounge access that goes unused

    return max(0.0, min(score, 1.0))


EXPERTS: tuple[Expert, ...] = (
    Expert("cashback", "Cashback Expert",
           "Direct, unconditional reward earning beats conditional perks.",
           _cashback_score),
    Expert("travel", "Travel Expert",
           "Lounge access and forex terms dominate for anyone who travels.",
           _travel_score),
    Expert("premium", "Premium Expert",
           "The largest total benefit wins, provided the fee is affordable.",
           _premium_score),
    Expert("cost_optimizer", "Cost Optimizer",
           "The best return per rupee of cost is the only fair comparison.",
           _cost_optimizer_score),
    Expert("risk", "Risk Agent",
           "Argues against any card whose case does not survive the user's "
           "actual behaviour and budget.",
           _risk_score),
)

DEBATE_SYSTEM = (
    "You are a credit card expert in a panel debate. Argue your assigned thesis "
    "in at most two sentences, using only the figures given. Never invent a "
    "number. Be direct."
)


def run_expert(
    expert: Expert, evaluations: list[dict[str, Any]], twin: dict[str, Any]
) -> dict[str, Any]:
    """Score every card from one expert's viewpoint and name their pick."""
    scored = [
        {
            "card_name": ev["card_name"],
            "score": round(float(expert.score(ev, twin)), 4),
            "net_annual_value": ev["net_annual_value"],
        }
        for ev in evaluations
    ]
    scored.sort(key=lambda s: s["score"], reverse=True)

    # The Risk Agent scores safety, so its "pick" is the card it objects to
    # least; its objections are the useful output.
    objections: list[dict[str, Any]] = []
    if expert.key == "risk":
        objections = [s for s in scored if s["score"] < 0.6]

    return {
        "expert": expert.key,
        "label": expert.label,
        "thesis": expert.thesis,
        "recommends": scored[0]["card_name"] if scored else None,
        "scores": scored,
        "objections": objections,
    }


def explain_expert(
    verdict: dict[str, Any],
    evaluation: dict[str, Any],
    provider: llm.Provider = "groq",
) -> str:
    """
    Phrase one expert's argument. Falls back to a deterministic sentence when no
    provider is available, so the debate always has readable output.
    """
    fallback = (
        f"{verdict['label']} recommends {verdict['recommends']}: "
        f"net realised value Rs {evaluation['net_annual_value']:,.0f} a year."
    )
    prompt = (
        f"Your thesis: {verdict['thesis']}\n"
        f"You recommend: {verdict['recommends']}\n"
        f"Net annual value: Rs {evaluation['net_annual_value']:,.0f}\n"
        f"Rewards: Rs {evaluation['rewards']['annual_rewards']:,.0f}, "
        f"lounge: Rs {evaluation['lounge']['lounge_value']:,.0f}, "
        f"cost: Rs {evaluation['cost']['cost']:,.0f}\n"
        "Argue for this card."
    )
    try:
        text = llm.chat(prompt, provider=provider, system=DEBATE_SYSTEM)
    except Exception:
        return fallback
    return (text or "").strip() or fallback


def deliberate(
    evaluations: list[dict[str, Any]],
    twin: dict[str, Any],
    with_arguments: bool = False,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Run the full panel. Pure unless `with_arguments=True`, which adds LLM prose.

    Returns a per-card consensus in 0..1 that Tier 5 consumes directly.
    """
    if not evaluations:
        return {
            "status": "no cards to deliberate",
            "verdicts": [], "consensus": {}, "objections": [],
        }

    by_name = {ev["card_name"]: ev for ev in evaluations}
    verdicts = [run_expert(e, evaluations, twin) for e in EXPERTS]

    if with_arguments:
        for verdict in verdicts:
            pick = by_name.get(verdict["recommends"])
            if pick:
                verdict["argument"] = explain_expert(verdict, pick, provider)

    # Weighted mean of each expert's score for each card.
    consensus: dict[str, float] = {}
    total_weight = sum(EXPERT_WEIGHT[e.key] for e in EXPERTS)
    for name in by_name:
        weighted = sum(
            EXPERT_WEIGHT[v["expert"]]
            * next(s["score"] for s in v["scores"] if s["card_name"] == name)
            for v in verdicts
        )
        consensus[name] = round(weighted / total_weight, 4)

    picks = [v["recommends"] for v in verdicts if v["expert"] != "risk"]
    agreement = round(
        max(picks.count(p) for p in set(picks)) / len(picks), 4
    ) if picks else 0.0

    risk_verdict = next(v for v in verdicts if v["expert"] == "risk")

    return {
        "status": "deliberated",
        "verdicts": verdicts,
        "consensus": consensus,
        "agreement": agreement,
        "unanimous": agreement == 1.0,
        "objections": risk_verdict["objections"],
        "contested": sorted(set(picks)),
    }
