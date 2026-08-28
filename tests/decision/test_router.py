"""
Decision Layer -> Intent Router.

Every test runs with `use_llm=False`, so the deterministic rule layer is fully
exercised offline. The LLM fallback is tested with a stub.
"""

from __future__ import annotations

import json

import pytest

import ml.src.common.llm as llm_module
from ml.src.decision.router import (
    INTENT_ROUTES,
    agents_for,
    classify_by_llm,
    classify_by_rules,
    route,
    router_node,
)
from ml.src.schemas.state import COUNCIL_AGENTS, new_state


@pytest.mark.parametrize(
    "query,expected",
    [
        ("is Doubler Capital a scam?", "fraud_check"),
        ("this offer promises guaranteed returns, is it legit?", "fraud_check"),
        ("which credit card should I get for online shopping?", "credit_card"),
        ("best cashback card for fuel?", "credit_card"),
        ("am I eligible for any government schemes?", "scheme_check"),
        ("can I get PM-KISAN?", "scheme_check"),
        ("my salary just got credited, what should I do?", "salary_event"),
        ("payday today, plan my month", "salary_event"),
        ("we're getting married next year", "life_event"),
        ("I just lost my job", "life_event"),
        ("how much will I have in 6 months?", "future_income"),
        ("what's my cash flow looking like", "future_income"),
        ("I want to save 20 lakh for a house down payment", "goal_planning"),
        ("when can I retire?", "goal_planning"),
        ("do I have enough emergency fund and insurance?", "safety_review"),
        ("give me a full financial review", "financial_planning"),
        ("where do I stand financially?", "financial_planning"),
    ],
)
def test_rule_based_routing(query, expected):
    assert route(query, use_llm=False)["intent"] == expected


def test_unmatched_query_falls_back_to_general():
    decision = route("what's the weather like", use_llm=False)
    assert decision["intent"] == "general"
    assert decision["method"] == "fallback"


def test_empty_query_is_general():
    assert route("", use_llm=False)["intent"] == "general"
    assert route("   ", use_llm=False)["intent"] == "general"


def test_rules_report_what_matched():
    intent, reason = classify_by_rules("is this a scam?")
    assert intent == "fraud_check"
    assert "scam" in reason.lower()


# --------------------------------------------------------------------------- #
# The routing table
# --------------------------------------------------------------------------- #

def test_fraud_check_activates_only_the_fraud_agent():
    """
    A routine review must never activate the network-bound fraud agent, and a
    fraud question must not activate eighteen agents.
    """
    assert route("is this a scam?", use_llm=False)["agents"] == ["fraud"]


def test_review_activates_every_agent():
    decision = route("give me a full financial review", use_llm=False)
    assert len(decision["agents"]) == sum(len(a) for a in COUNCIL_AGENTS.values())


def test_narrow_intents_stay_narrow():
    for query in ("which credit card?", "am I eligible for schemes?"):
        assert len(route(query, use_llm=False)["agents"]) <= 2


def test_every_intent_has_a_route():
    from ml.src.schemas.state import Intent
    import typing

    for intent in typing.get_args(Intent):
        assert intent in INTENT_ROUTES


def test_every_routed_agent_belongs_to_a_real_council():
    known = {a for agents in COUNCIL_AGENTS.values() for a in agents}
    for intent in INTENT_ROUTES:
        assert set(agents_for(intent)) <= known, intent


def test_routed_councils_are_valid():
    for intent, spec in INTENT_ROUTES.items():
        assert set(spec["councils"]) <= set(COUNCIL_AGENTS), intent


def test_simulation_and_deliberation_flags():
    assert route("how much will I have in 6 months?", use_llm=False)["requires_simulation"]
    assert route("we're getting married", use_llm=False)["requires_deliberation"]
    assert not route("is this a scam?", use_llm=False)["requires_deliberation"]


# --------------------------------------------------------------------------- #
# LLM fallback
# --------------------------------------------------------------------------- #

def test_llm_is_only_consulted_when_rules_miss(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: calls.append(p) or "general")
    route("is this a scam?", use_llm=True)
    assert calls == []


def test_llm_classifies_when_rules_miss(monkeypatch):
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "goal_planning")
    decision = route("thinking about the future a bit", use_llm=True)
    assert decision["intent"] == "goal_planning"
    assert decision["method"] == "llm"


def test_hallucinated_intent_is_rejected(monkeypatch):
    """An unknown label must degrade to `general`, not crash a workflow."""
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "buy_crypto_now")
    decision = route("something ambiguous entirely", use_llm=True)
    assert decision["intent"] == "general"
    assert "unknown intent" in decision["reason"]


def test_classifier_failure_is_survivable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_module, "chat", boom)
    decision = route("something ambiguous entirely", use_llm=True)
    assert decision["intent"] == "general"
    assert "unavailable" in decision["reason"]


def test_llm_answer_is_normalised(monkeypatch):
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "  Fraud Check.\n")
    assert classify_by_llm("x")[0] == "fraud_check"


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_routing_into_state(salaried_profile):
    state = new_state(salaried_profile, query="is Doubler Capital a scam?")
    patch = router_node(state, use_llm=False)
    assert set(patch) == {"intent", "routed_councils", "routing"}
    assert patch["intent"] == "fraud_check"
    assert patch["routed_councils"] == ["risk"]
    json.dumps(patch)
