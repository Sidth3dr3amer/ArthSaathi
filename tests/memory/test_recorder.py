"""
Memory Layer -> the "Update Memory" step.

Runs against `InMemoryStore` so it stays offline; the store contract itself is
verified against live Neon in `test_store.py`.
"""

from __future__ import annotations

import pytest

from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.memory.recorder import (
    AGENT_MEMORY_TYPE,
    memory_recall_node,
    memory_write_node,
    summarise,
)
from ml.src.memory.store import InMemoryStore
from ml.src.schemas.state import RESULT_KEYS, new_state


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def run_state(salaried_profile):
    """A state carrying one real agent result."""
    state = new_state(salaried_profile, query="am I covered for an emergency?")
    state.update(emergency_fund_node(state))
    return state


# --------------------------------------------------------------------------- #
# Routing table
# --------------------------------------------------------------------------- #

def test_every_result_key_has_a_memory_type():
    missing = [k for k in RESULT_KEYS if k not in AGENT_MEMORY_TYPE]
    assert missing == [], f"agents with no memory routing: {missing}"


def test_routing_uses_only_valid_memory_types():
    from ml.src.memory.models import MEMORY_TYPES

    assert set(AGENT_MEMORY_TYPE.values()) <= set(MEMORY_TYPES)


def test_behavioural_agents_write_behavioural_memory():
    for key in ("bias_detection_result", "habit_formation_result",
                "nudge_strategy_result", "literacy_result"):
        assert AGENT_MEMORY_TYPE[key] == "behavioral"


# --------------------------------------------------------------------------- #
# Summarisation
# --------------------------------------------------------------------------- #

def test_emergency_fund_summary_is_a_readable_sentence(run_state):
    text = summarise("emergency_fund_result", run_state["emergency_fund_result"])
    assert "Emergency fund is" in text
    assert "%" in text
    assert len(text) < 300


def test_summariser_falls_back_when_the_payload_is_unexpected():
    """A malformed result must still produce something embeddable, not raise."""
    text = summarise("emergency_fund_result", {"unexpected": True})
    assert isinstance(text, str) and text


def test_unknown_agent_gets_a_generic_summary():
    text = summarise("something_new_result", {"a": 1})
    assert "something new" in text


def test_summary_is_length_capped():
    huge = {"blob": "x" * 5_000}
    assert len(summarise("unmapped_result", huge)) <= 600


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def test_write_persists_one_memory_per_agent_result(run_state, store):
    patch = memory_write_node(run_state, store=store)
    assert patch["memory_written"] is True
    stored = store.recent(run_state["user_id"])
    assert len(stored) == 1
    assert stored[0]["memory_type"] == "semantic"
    assert stored[0]["source_agent"] == "emergency_fund"


def test_write_stores_the_full_result_as_payload(run_state, store):
    memory_write_node(run_state, store=store)
    payload = store.recent(run_state["user_id"])[0]["payload"]
    assert payload == run_state["emergency_fund_result"]


def test_final_decision_is_remembered_with_high_importance(run_state, store):
    run_state["final_decision"] = "Prioritise the emergency fund for three months."
    memory_write_node(run_state, store=store)
    judge = [m for m in store.recent(run_state["user_id"]) if m["source_agent"] == "judge"]
    assert len(judge) == 1
    assert judge[0]["importance"] == 0.9
    assert "Prioritise the emergency fund" in judge[0]["content"]


def test_write_on_an_empty_state_records_nothing(salaried_profile, store):
    patch = memory_write_node(new_state(salaried_profile), store=store)
    assert patch["memory_written"] is False
    assert store.recent("test-salaried") == []


def test_a_failing_store_does_not_break_the_workflow(run_state):
    """A memory outage must not abort a recommendation the user is waiting on."""

    class Broken(InMemoryStore):
        def remember(self, *a, **k):
            raise RuntimeError("neon unreachable")

    patch = memory_write_node(run_state, store=Broken())
    assert patch["memory_written"] is False
    assert any("neon unreachable" in e for e in patch["errors"])


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #

def test_recall_returns_prior_memories_in_a_fresh_session(run_state, store, salaried_profile):
    memory_write_node(run_state, store=store)

    fresh = new_state(salaried_profile, query="how is my emergency fund doing?")
    patch = memory_recall_node(fresh, store=store)
    assert len(patch["recalled_memories"]) == 1
    assert "Emergency fund is" in patch["recalled_memories"][0]["content"]


def test_recall_ranks_the_relevant_memory_first(store, salaried_profile):
    uid = salaried_profile.user_id
    store.remember(uid, "goal", "Saving for a home down payment of 15 lakh.")
    store.remember(uid, "semantic", "Emergency fund is 11% funded and critical.")
    store.remember(uid, "behavioral", "Overspends on dining at month end.")

    state = new_state(salaried_profile, query="tell me about my emergency fund")
    hits = memory_recall_node(state, store=store)["recalled_memories"]
    assert "Emergency fund" in hits[0]["content"]


def test_recall_without_a_query_falls_back_to_recency(store, salaried_profile):
    uid = salaried_profile.user_id
    store.remember(uid, "goal", "first")
    store.remember(uid, "goal", "second")

    state = new_state(salaried_profile)          # no query
    hits = memory_recall_node(state, store=store)["recalled_memories"]
    assert hits[0]["content"] == "second"


def test_recall_can_be_scoped_to_types(store, salaried_profile):
    uid = salaried_profile.user_id
    store.remember(uid, "goal", "goal memory")
    store.remember(uid, "behavioral", "behavioural memory")

    state = new_state(salaried_profile, query="anything")
    hits = memory_recall_node(state, store=store, memory_types=["behavioral"])["recalled_memories"]
    assert {h["memory_type"] for h in hits} == {"behavioral"}


def test_recall_respects_the_limit(store, salaried_profile):
    for i in range(10):
        store.remember(salaried_profile.user_id, "episodic", f"memory {i}")
    state = new_state(salaried_profile, query="memory")
    assert len(memory_recall_node(state, store=store, limit=3)["recalled_memories"]) == 3


def test_recall_survives_a_store_outage(salaried_profile):
    class Broken(InMemoryStore):
        def recall(self, *a, **k):
            raise RuntimeError("neon unreachable")

    state = new_state(salaried_profile, query="anything")
    patch = memory_recall_node(state, store=Broken())
    assert patch["recalled_memories"] == []
    assert any("neon unreachable" in e for e in patch["errors"])


def test_recall_on_a_new_user_is_empty(store, salaried_profile):
    state = new_state(salaried_profile, query="anything at all")
    assert memory_recall_node(state, store=store)["recalled_memories"] == []


# --------------------------------------------------------------------------- #
# Full loop
# --------------------------------------------------------------------------- #

def test_write_then_recall_round_trip(run_state, store, salaried_profile):
    """The Day-2 acceptance criterion: an agent writes, a later run recalls."""
    run_state["final_decision"] = "Build three months of runway before investing."
    memory_write_node(run_state, store=store)

    later = new_state(salaried_profile, query="what did we decide about runway?")
    recalled = memory_recall_node(later, store=store)["recalled_memories"]
    assert any("Build three months of runway" in m["content"] for m in recalled)
