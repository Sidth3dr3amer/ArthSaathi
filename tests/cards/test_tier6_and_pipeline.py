"""
Credit Card Tier 6 -> Explanation Agent, and the full Tier 1-6 pipeline.

Golden values are pinned against the four curated cards in
`CreditCardDataMaker_Final/final_decision/`, so a silent change to any tier's
arithmetic fails here rather than in a demo.
"""

from __future__ import annotations

import json

import pytest

import ml.src.common.llm as llm_module
from ml.src.cards.pipeline import card_intelligence_node, run_card_intelligence
from ml.src.cards.tier5_ranking import rank_cards
from ml.src.cards.tier6_explain import build_explanation, explain
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


@pytest.fixture
def ranking(evaluations, simulations, deliberation, engine_bits, tier1):
    _, _, eligible, _ = engine_bits
    return rank_cards(evaluations, simulations, deliberation, eligible,
                      tier1["profiler"], tier1["spending"], tier1["twin"])


@pytest.fixture
def explanation(ranking, evaluations, simulations, deliberation):
    return build_explanation(ranking, evaluations, simulations, deliberation)


# =========================================================================== #
# Tier 6 -- Explanation
# =========================================================================== #

def test_the_explanation_follows_the_docs_numbered_form(explanation):
    assert explanation["status"] == "explained"
    assert len(explanation["points"]) >= 4
    assert explanation["text"].startswith("1. ")


def test_it_states_annual_reward_earnings(explanation):
    assert "earns about Rs" in explanation["points"][0]


def test_it_states_lounge_visits_used_against_offered(explanation):
    """The doc's example is 'You will use 9 of 12 lounge visits.'"""
    assert any("lounge visits" in p and " of " in p for p in explanation["points"])


def test_it_states_the_net_benefit_after_the_fee(explanation):
    assert any("Net expected benefit" in p for p in explanation["points"])


def test_it_compares_against_the_runner_up(explanation):
    assert any("Better than" in p for p in explanation["points"])


def test_it_gives_the_honest_range_not_just_the_average(explanation):
    assert any("bad year" in p and "good year" in p for p in explanation["points"])


def test_it_names_the_realisation_gap(explanation):
    """The number card marketing omits: advertised value the user will not collect."""
    assert any("advertises about Rs" in p for p in explanation["points"])


def test_it_surfaces_a_risk_objection_when_there_is_one(
    evaluations, simulations, engine_bits, tier1
):
    _, _, eligible, _ = engine_bits
    objecting = {
        "consensus": {e["card_name"]: 0.5 for e in evaluations},
        "objections": [{"card_name": evaluations[0]["card_name"], "score": 0.2}],
        "agreement": 0.5, "contested": [], "verdicts": [],
    }
    ranked = rank_cards(evaluations, simulations, objecting, eligible,
                        tier1["profiler"], tier1["spending"], tier1["twin"])
    out = build_explanation(ranked, evaluations, simulations, objecting)
    if out["card_name"] == evaluations[0]["card_name"]:
        assert out["has_objection"] is True
        assert any("One caution" in p for p in out["points"])


def test_nothing_to_rank_degrades_cleanly():
    out = build_explanation({"status": "no cards to rank", "ranked": []}, [], [], {})
    assert out["status"] == "nothing to explain"
    assert out["points"] == []


def test_deterministic_mode_returns_the_numbered_draft(explanation):
    out = explain(explanation, use_llm=False)
    assert out["method"] == "deterministic"
    assert out["prose"] == explanation["text"]


def test_the_llm_only_rewrites_it_never_computes(explanation, monkeypatch):
    seen = {}
    monkeypatch.setattr(llm_module, "chat",
                        lambda p, system=None, **k: seen.update(
                            {"prompt": p, "system": system}) or "rewritten")
    out = explain(explanation, use_llm=True)
    assert out["prose"] == "rewritten"
    assert seen["prompt"] == explanation["text"]
    assert "never add a number" in seen["system"]


def test_a_provider_outage_falls_back_to_the_draft(explanation, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(llm_module, "chat", boom)
    out = explain(explanation, use_llm=True)
    assert out["prose"] == explanation["text"]
    assert out["method"].startswith("fallback")


def test_explanation_is_json_serialisable(explanation):
    json.dumps(explanation)


# =========================================================================== #
# Full pipeline
# =========================================================================== #

def test_the_pipeline_runs_all_six_tiers(card_user):
    out = run_card_intelligence(card_user)
    assert out["status"] == "complete"
    assert set(out["tiers"]) == {"tier1", "tier2", "tier3", "tier4", "tier5", "tier6"}


def test_the_pipeline_recommends_a_card_with_an_explanation(card_user):
    out = run_card_intelligence(card_user)
    recommendation = out["recommendation"]
    assert recommendation["card_name"]
    assert recommendation["net_annual_value"] > 0
    assert 0 <= recommendation["final_score_percent"] <= 100
    assert len(recommendation["points"]) >= 4


def test_golden_winner_on_the_curated_database(card_user):
    """Pinned so a change to any tier's arithmetic is deliberate, not incidental."""
    out = run_card_intelligence(card_user)
    assert out["cards_considered"] == 4
    assert out["eligible"] == 4
    assert out["recommendation"]["card_name"] == "Axis Bank Atlas Credit Card"
    assert out["recommendation"]["net_annual_value"] == pytest.approx(28_750, abs=1)


def test_golden_component_breakdown(card_user):
    top = run_card_intelligence(card_user)["top_cards"][0]
    components = top["components"]
    assert components["net_annual_value"] == pytest.approx(1.0)
    assert components["approval_probability"] == pytest.approx(0.70, abs=0.01)
    assert top["final_score_percent"] == pytest.approx(71.5, abs=0.5)


def test_the_panel_is_split_on_this_user(card_user):
    tier4 = run_card_intelligence(card_user)["tiers"]["tier4"]
    assert tier4["unanimous"] is False
    assert len(tier4["contested"]) == 2


def test_a_revolver_gets_a_different_answer(revolver):
    """
    Rewards are discounted to zero for someone paying 42%, so the ranking should
    not be driven by reward earning.
    """
    out = run_card_intelligence(revolver)
    if out["status"] == "complete":
        for evaluation in out["tiers"]["tier2"]:
            assert evaluation["rewards"]["annual_rewards"] == 0
        assert out["tiers"]["tier1"]["twin"]["rewards_are_real"] is False


def test_an_empty_database_degrades_rather_than_raising(card_user):
    out = run_card_intelligence(card_user, cards=[])
    assert out["status"] == "no card database"
    assert out["recommendation"] is None


def test_no_eligible_cards_explains_why():
    """max_annual_fee defaults to 0, which filters out every fee-charging card."""
    out = run_card_intelligence(UserProfile(user_id="strict", age=30,
                                            monthly_income=60_000))
    assert out["status"] == "no eligible cards"
    assert "max_annual_fee" in out["reason"]
    assert out["recommendation"] is None


def test_the_pipeline_is_deterministic_without_an_llm(card_user):
    assert run_card_intelligence(card_user) == run_card_intelligence(card_user)


def test_the_pipeline_is_json_serialisable(card_user):
    json.dumps(run_card_intelligence(card_user))


def test_the_pipeline_does_not_mutate_the_profile(card_user):
    before = card_user.model_dump()
    run_card_intelligence(card_user)
    assert card_user.model_dump() == before


def test_top_n_is_respected(card_user):
    assert len(run_card_intelligence(card_user, top_n=2)["top_cards"]) == 2


def test_node_writes_one_key_and_serialises(card_user):
    patch = card_intelligence_node(new_state(card_user))
    assert set(patch) == {"card_intelligence_result"}
    json.dumps(patch)


def test_node_passes_transactions_through(card_user):
    state = new_state(card_user)
    state["transactions"] = [
        {"date": "2026-01-05", "amount": 4_000, "category": "dining", "direction": "debit"},
    ]
    result = card_intelligence_node(state)["card_intelligence_result"]
    assert "observed" in result["tiers"]["tier1"]["spending"]["source"]
