"""Credit Card Tier 4 -> the five-expert deliberation panel."""

from __future__ import annotations

import json

import pytest

import ml.src.common.llm as llm_module
from ml.src.cards.tier1_profiler import run_tier1
from ml.src.cards.tier2_evaluation import evaluate_all
from ml.src.cards.tier4_experts import (
    EXPERTS,
    EXPERT_WEIGHT,
    RECURRING_REWARD_COMPONENTS,
    deliberate,
    explain_expert,
    run_expert,
)
from ml.src.councils.growth.credit_card import (
    analyze_spend_profile,
    filter_eligible_cards,
    profile_to_engine_dict,
)


def test_all_five_experts_from_the_doc_are_present():
    assert {e.key for e in EXPERTS} == {
        "cashback", "travel", "premium", "cost_optimizer", "risk",
    }


def test_every_expert_has_a_thesis_and_a_weight():
    for expert in EXPERTS:
        assert len(expert.thesis) > 20
        assert expert.key in EXPERT_WEIGHT


def test_the_risk_agent_carries_a_double_vote():
    """A veto is more informative than an endorsement."""
    assert EXPERT_WEIGHT["risk"] == 2.0
    assert all(v == 1.0 for k, v in EXPERT_WEIGHT.items() if k != "risk")


# --------------------------------------------------------------------------- #
# Individual experts
# --------------------------------------------------------------------------- #

def test_the_cashback_expert_picks_a_cashback_card(evaluations, tier1):
    """
    Its thesis is direct, unconditional earning. Counting a one-time welcome
    bonus or a conditional milestone benefit would let a premium travel card
    win this vote, which contradicts the thesis.
    """
    verdict = run_expert(EXPERTS[0], evaluations, tier1["twin"])
    pick = next(e for e in evaluations if e["card_name"] == verdict["recommends"])
    assert pick["card_type"].upper() == "CASHBACK"


def test_the_cashback_expert_ignores_one_time_and_conditional_value():
    assert set(RECURRING_REWARD_COMPONENTS) == {"base_rewards", "utility_bonus"}


def test_the_travel_expert_favours_lounge_access_for_a_traveller(
    card_db, frequent_flyer
):
    twin = run_tier1(frequent_flyer)["twin"]
    engine_profile = profile_to_engine_dict(frequent_flyer)
    spend = analyze_spend_profile(engine_profile)
    eligible, _ = filter_eligible_cards(engine_profile, card_db)
    evaluated = evaluate_all(eligible, engine_profile, spend, twin)

    verdict = run_expert(next(e for e in EXPERTS if e.key == "travel"), evaluated, twin)
    pick = next(e for e in evaluated if e["card_name"] == verdict["recommends"])
    assert pick["lounge"]["lounge_value"] > 0


def test_the_premium_expert_stands_down_when_a_fee_is_unaffordable(evaluations):
    poor_twin = {"can_absorb_premium_fee": False, "annual_spend": 100_000,
                 "travel_profile": "low", "fee_tolerance": 0}
    verdict = run_expert(next(e for e in EXPERTS if e.key == "premium"),
                         evaluations, poor_twin)
    assert all(s["score"] <= 0.1 for s in verdict["scores"])


def test_the_risk_agent_objects_when_a_fee_exceeds_tolerance(evaluations):
    broke_twin = {"fee_tolerance": 0, "rewards_are_real": True,
                  "approval_headroom": True, "annual_spend": 100_000,
                  "travel_profile": "low", "can_absorb_premium_fee": False}
    verdict = run_expert(next(e for e in EXPERTS if e.key == "risk"),
                         evaluations, broke_twin)
    assert verdict["objections"]


def test_the_risk_agent_objects_for_a_revolver(evaluations, revolver):
    twin = run_tier1(revolver)["twin"]
    verdict = run_expert(next(e for e in EXPERTS if e.key == "risk"), evaluations, twin)
    assert verdict["objections"]


def test_a_healthy_user_draws_no_objections(evaluations, tier1):
    verdict = run_expert(next(e for e in EXPERTS if e.key == "risk"),
                         evaluations, tier1["twin"])
    assert verdict["objections"] == []


def test_every_expert_scores_every_card(evaluations, tier1):
    for expert in EXPERTS:
        verdict = run_expert(expert, evaluations, tier1["twin"])
        assert len(verdict["scores"]) == len(evaluations)
        assert all(0.0 <= s["score"] <= 1.0 for s in verdict["scores"])


def test_scores_are_ranked_within_each_expert(evaluations, tier1):
    for expert in EXPERTS:
        scores = [s["score"] for s in run_expert(expert, evaluations, tier1["twin"])["scores"]]
        assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #

def test_deliberation_produces_a_consensus_per_card(evaluations, deliberation):
    assert set(deliberation["consensus"]) == {e["card_name"] for e in evaluations}
    assert all(0.0 <= v <= 1.0 for v in deliberation["consensus"].values())


def test_disagreement_is_reported_as_the_signal(deliberation):
    """A contested pick is exactly what Tier 5's AgentConsensus consumes."""
    assert 0.0 <= deliberation["agreement"] <= 1.0
    assert len(deliberation["contested"]) >= 1


def test_the_panel_is_genuinely_split_on_the_sample_user(deliberation):
    assert deliberation["unanimous"] is False
    assert len(deliberation["contested"]) == 2


def test_five_verdicts_are_returned(deliberation):
    assert len(deliberation["verdicts"]) == 5


def test_no_cards_degrades_rather_than_raising(tier1):
    out = deliberate([], tier1["twin"])
    assert out["status"] == "no cards to deliberate"
    assert out["consensus"] == {}


def test_deliberation_is_deterministic_without_an_llm(evaluations, tier1):
    assert deliberate(evaluations, tier1["twin"]) == deliberate(evaluations, tier1["twin"])


def test_deliberation_is_json_serialisable(deliberation):
    json.dumps(deliberation)


# --------------------------------------------------------------------------- #
# LLM arguments
# --------------------------------------------------------------------------- #

def test_arguments_are_only_added_when_asked(evaluations, tier1, monkeypatch):
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "argued")
    plain = deliberate(evaluations, tier1["twin"], with_arguments=False)
    assert all("argument" not in v for v in plain["verdicts"])

    argued = deliberate(evaluations, tier1["twin"], with_arguments=True)
    assert all(v["argument"] == "argued" for v in argued["verdicts"])


def test_an_argument_falls_back_when_the_provider_is_down(evaluations, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(llm_module, "chat", boom)
    verdict = {"label": "Cashback Expert", "recommends": evaluations[0]["card_name"],
               "thesis": "t"}
    text = explain_expert(verdict, evaluations[0])
    assert "Cashback Expert recommends" in text


def test_arguments_never_invent_numbers(evaluations, tier1, monkeypatch):
    """The prompt carries the figures; the model is told not to add any."""
    seen = {}
    monkeypatch.setattr(llm_module, "chat",
                        lambda p, system=None, **k: seen.update(
                            {"prompt": p, "system": system}) or "ok")
    deliberate(evaluations, tier1["twin"], with_arguments=True)
    assert "Never invent a number" in seen["system"]
    assert "Net annual value" in seen["prompt"]
