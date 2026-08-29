"""
FastAPI backend.

Every test runs OFFLINE: the LLM is stubbed and the store is forced in-memory,
including during app startup -- otherwise the lifespan hook would open a real
Neon connection on every test session.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import ml.src.common.llm as llm_module
import server.main as server_main
from ml.src.memory.store import InMemoryStore, set_store
from ml.src.workflows.catalogue import WORKFLOWS

PROFILE = {
    "user_id": "api-test",
    "age": 33,
    "job_type": "salaried",
    "occupation": "farmer",
    "residence": "rural",
    "land_holding_ha": 1.2,
    "monthly_income": 95_000,
    "essential_expenses": 48_000,
    "existing_emergency_fund": 40_000,
    "retirement_corpus": 300_000,
    "dependents": 2,
    "has_health_insurance": False,
    "annual_household_income": 1_140_000,
    "max_annual_fee": 3_000,
    "monthly_spend": {"rent": 26_000, "dining": 8_000, "groceries": 8_000},
    "debts": [{"name": "Card", "debt_type": "credit_card",
               "outstanding_amount": 160_000, "interest_rate": 42.0,
               "minimum_due": 8_000}],
}


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def client(monkeypatch, store):
    """A TestClient with no network and no database."""
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "[stubbed]")

    def _init_store():
        set_store(store)
        return store

    # The lifespan hook would otherwise connect to Neon on startup.
    monkeypatch.setattr(server_main, "init_store", _init_store)
    with TestClient(server_main.app) as c:
        yield c


def chat(client, message, **extra):
    body = {"user_id": "api-test", "message": message,
            "profile": PROFILE, "use_llm_router": False}
    body.update(extra)
    return client.post("/chat", json=body)


# =========================================================================== #
# Meta
# =========================================================================== #

def test_health_reports_wiring(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agents"] == 18
    assert body["workflows"] == len(WORKFLOWS)
    # Derived, not hardcoded, so adding a provider does not break this.
    from ml.src.common.llm import _KEY_FOR
    assert set(body["providers"]) == set(_KEY_FOR)


def test_health_never_leaks_key_values(client):
    """Only booleans -- a health endpoint is often public."""
    raw = json.dumps(client.get("/health").json())
    assert "gsk_" not in raw and "npg_" not in raw
    assert all(isinstance(v, bool) for v in client.get("/health").json()["providers"].values())


def test_index_lists_every_endpoint(client):
    endpoints = client.get("/").json()["endpoints"]
    assert "/chat" in endpoints and "/workflows" in endpoints
    assert len(endpoints) >= 15


def test_openapi_is_served(client):
    assert len(client.get("/openapi.json").json()["paths"]) >= 15


# =========================================================================== #
# /chat
# =========================================================================== #

@pytest.mark.parametrize(
    "message,workflow",
    [
        ("which credit card should I get?", "Credit Card Advisory"),
        ("is Doubler Capital a scam?", "Fraud Protection"),
        ("am I eligible for government schemes?", "Benefits / Scheme Council"),
        ("how much will I have in 6 months?", "Future Income Simulation"),
        ("give me a full financial review", "Full Financial Review"),
    ],
)
def test_chat_routes_to_the_right_workflow(client, message, workflow):
    body = chat(client, message).json()
    assert body["workflow"] == workflow
    assert body["errors"] == []


def test_chat_writes_memory(client, store):
    assert chat(client, "give me a full financial review").json()["memory_written"] is True
    assert store.recent("api-test")


def test_chat_recommendations_are_display_strings(client):
    """
    The credit-card agent returns card OBJECTS in its `recommendations`; the
    route must reduce them to lines rather than dumping the card database.
    """
    body = chat(client, "which credit card should I get?").json()
    lines = body["recommendations"]["credit_card"]

    # Assert the behaviour, not the phrasing: every entry is a readable line
    # naming a card, and the payload stays compact.
    assert lines
    assert all(isinstance(line, str) for line in lines)
    assert all("Credit Card" in line for line in lines)
    assert any("Rs" in line for line in lines)
    assert len(json.dumps(body)) < 20_000


def test_chat_full_review_runs_every_agent_and_deliberates(client):
    body = chat(client, "give me a full financial review").json()
    assert len(body["agents_run"]) == 18
    assert len(body["council_verdicts"]) == 5
    assert body["allocation_plan"]


def test_chat_uses_the_stored_profile_when_none_is_supplied(client):
    client.put("/profile/api-test", json=PROFILE)
    body = client.post("/chat", json={"user_id": "api-test", "message": "how am I doing?",
                                      "use_llm_router": False}).json()
    assert body["profile_source"] == "stored"


def test_chat_for_an_unknown_user_starts_a_new_profile(client):
    body = client.post("/chat", json={"user_id": "brand-new", "message": "hello there",
                                      "use_llm_router": False}).json()
    assert body["profile_source"] == "new"
    assert body["errors"] == []


def test_chat_rejects_an_empty_message(client):
    assert client.post("/chat", json={"user_id": "x", "message": ""}).status_code == 422


def test_chat_requires_a_user_id(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


def test_chat_degrades_instead_of_500(client, monkeypatch):
    """An agent-layer failure must not surface as a stack trace."""
    import server.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    response = chat(client, "give me a full financial review")
    assert response.status_code == 200
    assert response.json()["workflow"] == "unavailable"
    assert any("boom" in e for e in response.json()["errors"])


def test_an_inline_profile_cannot_write_into_another_users_record(client):
    """The body's user_id is authoritative over the inline profile's."""
    hostile = {**PROFILE, "user_id": "victim"}
    body = client.post("/chat", json={"user_id": "attacker", "message": "hello",
                                      "profile": hostile, "use_llm_router": False}).json()
    assert body["errors"] == []
    assert client.get("/profile/victim").status_code == 404


# =========================================================================== #
# /ask
# =========================================================================== #

def test_ask_answers_from_memory(client, store):
    """
    The query shares vocabulary with the memory on purpose: the default hashed
    embedding backend matches lexical overlap, not meaning, so "what do I earn?"
    would score below the retrieval floor against "monthly income".
    """
    store.remember("api-test", "semantic", "User monthly income is Rs 95,000.")
    body = client.post("/ask", json={"user_id": "api-test",
                                     "question": "what is my monthly income?"}).json()
    assert body["answer"] == "[stubbed]"
    assert body["memories_included"] >= 1


def test_ask_degrades_without_a_provider(client, monkeypatch):
    monkeypatch.setattr(llm_module, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    body = client.post("/ask", json={"user_id": "api-test", "question": "what do I earn?"}).json()
    assert "on record" in body["answer"]


# =========================================================================== #
# /profile
# =========================================================================== #

def test_profile_turn_extracts_and_asks_the_next_question(client, monkeypatch):
    monkeypatch.setattr(
        llm_module, "chat",
        lambda p, system=None, **k: ('{"monthly_income": 35000, "dependents": 3}'
                                     if system and "extract" in system.lower() else "ok"),
    )
    body = client.post("/profile/turn", json={
        "user_id": "onboard", "message": "I earn 35 thousand, 3 dependants"}).json()
    assert body["profile"]["monthly_income"] == 35_000
    assert "monthly_income" in body["stages"]["fields_applied"]
    assert body["next_question"]["field"] == "essential_expenses"


def test_profile_turn_reports_stage_progress(client, monkeypatch):
    monkeypatch.setattr(llm_module, "chat",
                        lambda p, system=None, **k: '{"monthly_income": 35000}'
                        if system and "extract" in system.lower() else "ok")
    stages = client.post("/profile/turn", json={
        "user_id": "onboard", "message": "I earn 35 thousand"}).json()["stages"]
    assert set(stages) >= {"language", "amounts_found", "fields_applied", "memories_written"}
    assert stages["amounts_found"] >= 1


def test_profile_turn_degrades_on_pipeline_failure(client, monkeypatch):
    import server.routes.profile as profile_routes

    monkeypatch.setattr(profile_routes, "run_profile_agent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.post("/profile/turn", json={"user_id": "x", "message": "hi"})
    assert response.status_code == 200
    assert "error" in response.json()["stages"]


def test_profile_put_then_get_round_trips(client):
    assert client.put("/profile/api-test", json=PROFILE).json()["saved"] is True
    body = client.get("/profile/api-test").json()
    assert body["profile"]["monthly_income"] == 95_000
    assert body["can_advise"] is True


def test_profile_get_404s_for_an_unknown_user(client):
    assert client.get("/profile/nobody-at-all").status_code == 404


def test_questions_work_for_a_brand_new_user(client):
    """A new user is exactly who needs the queue, so this must not 404."""
    body = client.get("/profile/brand-new/questions").json()
    assert body["next_question"]["field"] == "monthly_income"
    assert len(body["queue"]) == 7
    assert body["completeness"]["can_advise"] is False
    assert body["blocked_councils"]


def test_questions_shrink_as_the_profile_fills(client):
    client.put("/profile/api-test", json=PROFILE)
    body = client.get("/profile/api-test/questions").json()
    assert body["completeness"]["can_advise"] is True
    assert body["blocked_councils"] == []


def test_question_limit_is_respected(client):
    assert len(client.get("/profile/brand-new/questions?limit=3").json()["queue"]) == 3


# =========================================================================== #
# /workflows
# =========================================================================== #

def test_workflow_list_describes_all_nine(client):
    body = client.get("/workflows").json()
    assert body["count"] == len(WORKFLOWS)
    assert len(body["agents"]) == 18
    for wf in body["workflows"]:
        assert wf["label"] and wf["description"] and wf["agents"]


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_workflow_runs_directly(client, name):
    body = client.post(f"/workflow/{name}",
                       json={"user_id": "api-test", "query": "run", "profile": PROFILE}).json()
    assert body["workflow"] == name
    assert body["memory_written"] is True
    assert body["steps"][0] == "recall" and body["steps"][-1] == "remember"


def test_unknown_workflow_404s(client):
    response = client.post("/workflow/not-a-workflow",
                           json={"user_id": "api-test", "profile": PROFILE})
    assert response.status_code == 404
    assert "unknown workflow" in response.json()["detail"]


def test_workflow_returns_full_agent_results(client):
    body = client.post("/workflow/goal_planning",
                       json={"user_id": "api-test", "query": "goals",
                             "profile": PROFILE}).json()
    assert "emergency_fund_result" in body["results"]
    assert body["results"]["emergency_fund_result"]["status"]


def test_workflow_degrades_on_failure(client, monkeypatch):
    import server.routes.workflows as wf_routes

    monkeypatch.setattr(wf_routes, "get_workflow",
                        lambda n: type("G", (), {
                            "invoke": lambda self, s: (_ for _ in ()).throw(RuntimeError("boom")),
                            "workflow_steps": [],
                        })())
    response = client.post("/workflow/credit_card",
                           json={"user_id": "api-test", "profile": PROFILE})
    assert response.status_code == 200
    assert any("boom" in e for e in response.json()["errors"])


# =========================================================================== #
# /memory
# =========================================================================== #

def test_memory_types_are_listed(client):
    assert len(client.get("/memory/types").json()["types"]) == 6


def test_recent_and_recall(client, store):
    store.remember("api-test", "semantic", "User monthly income is Rs 95,000.")
    store.remember("api-test", "goal", "Saving for a home down payment.")

    recent = client.get("/memory/api-test").json()
    assert recent["count"] == 2

    recalled = client.post("/memory/api-test/recall",
                           json={"query": "what is my income?"}).json()
    assert recalled["memories"][0]["memory_type"] == "semantic"


def test_memory_can_be_filtered_by_type(client, store):
    store.remember("api-test", "semantic", "a fact")
    store.remember("api-test", "goal", "a goal")
    body = client.get("/memory/api-test?memory_type=goal").json()
    assert [m["memory_type"] for m in body["memories"]] == ["goal"]


def test_an_unknown_memory_type_is_422(client):
    assert client.get("/memory/api-test?memory_type=telepathic").status_code == 422
    assert client.post("/memory/api-test/recall",
                       json={"query": "x", "memory_types": ["nope"]}).status_code == 422


def test_recall_requires_a_query(client):
    assert client.post("/memory/api-test/recall", json={}).status_code == 422


def test_forget_removes_memories(client, store):
    store.remember("api-test", "semantic", "a fact")
    assert client.delete("/memory/api-test").json()["removed"] == 1
    assert client.get("/memory/api-test").json()["count"] == 0


def test_memory_for_an_unknown_user_is_empty_not_an_error(client):
    body = client.get("/memory/nobody").json()
    assert body["count"] == 0 and body["errors"] == []


def test_memory_degrades_on_store_outage(client, monkeypatch):
    import server.routes.memory as mem_routes

    class Broken(InMemoryStore):
        def recent(self, *a, **k):
            raise RuntimeError("neon down")

    monkeypatch.setattr(mem_routes, "get_store", lambda: Broken())
    response = client.get("/memory/api-test")
    assert response.status_code == 200
    assert any("neon down" in e for e in response.json()["errors"])


# =========================================================================== #
# /cards
# =========================================================================== #

def test_card_database_is_listed(client):
    body = client.get("/cards").json()
    assert all(c["card_name"] for c in body["cards"])
    # Hand-checked cards plus ones promoted out of the raw extraction. Each row
    # says which it is, so a consumer can tell a verified card from a parsed one.
    assert body["count"] == len(body["cards"]) > body["confirmed_count"] == 4
    assert all("eligibility_confirmed" in c for c in body["cards"])


def test_card_recommendation_is_ranked(client):
    body = client.post("/cards/recommend?top_n=3",
                       json={"user_id": "api-test", "profile": PROFILE}).json()
    values = [r["net_annual_value"] for r in body["recommendations"]]
    assert values == sorted(values, reverse=True)
    assert body["cards_considered"] >= 4


def test_a_revolving_user_is_warned_rather_than_sold_a_second_card(client):
    """
    Recommending a card to someone paying 42% on the one they hold is the
    failure this endpoint has to refuse to make.
    """
    # PROFILE already carries Rs 1,60,000 at 42%.
    body = client.post("/cards/recommend?top_n=3",
                       json={"user_id": "api-test", "profile": PROFILE}).json()
    assert body["recommend_new_card"] is False
    assert "1,60,000" in body["caution"] or "160,000" in body["caution"]
    assert body["existing_cards"]["annual_interest_cost"] == pytest.approx(67_200)


def test_a_user_with_no_card_balance_still_gets_recommendations(client):
    """The warning must not suppress advice for someone carrying nothing."""
    body = client.post("/cards/recommend?top_n=3",
                       json={"user_id": "api-test",
                             "profile": {**PROFILE, "debts": []}}).json()
    assert body["recommend_new_card"] is True
    assert body["caution"] is None
    assert body["recommendations"]


def test_spend_profile_returns_an_analysis(client):
    body = client.post("/cards/spend-profile",
                       json={"user_id": "api-test", "profile": PROFILE}).json()
    assert body["dominant_type"]
    assert body["total_annual"] > 0


# =========================================================================== #
# /voice
# =========================================================================== #

def test_voice_status_does_not_load_the_model(client):
    """
    Reporting status must stay cheap -- the module loads a ~500 MB Whisper model
    at import, so `/voice/status` deliberately does not import it.
    """
    body = client.get("/voice/status").json()
    assert body["module_present"] is True
    assert body["loaded"] is False
    assert "faster-whisper" in body["requires"]
