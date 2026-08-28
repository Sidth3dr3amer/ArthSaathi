"""
Memory Layer -> MemoryStore contract.

Every test here runs TWICE: once against `InMemoryStore` (fast, offline, always)
and once against `PostgresMemoryStore` on Neon (marked `live`, skipped when
DATABASE_URL is absent). A behaviour that differs between the two is a bug --
agents must not care which is configured.

Run offline only:   pytest -m "not live"
Run against Neon:   pytest -m live
"""

from __future__ import annotations

import uuid

import pytest

from ml.src.common import config
from ml.src.memory.models import MEMORY_TYPES
from ml.src.memory.store import (
    InMemoryStore,
    PostgresMemoryStore,
    get_store,
    set_store,
)

MEMORIES = [
    ("episodic", "User asked whether to prioritise the emergency fund over debt repayment.", "judge"),
    ("goal", "Saving for a home down payment of 15 lakh over 4 years.", "goal_allocation"),
    ("behavioral", "Consistently overspends on dining in the last week of every month.", "habit_formation"),
    ("simulation", "Monte Carlo projected a minimum balance of 146824 over six months.", "stability"),
]


def _pg_store():
    store = PostgresMemoryStore()
    store.create_schema()
    return store


@pytest.fixture(
    params=[
        pytest.param("memory", id="in_memory"),
        pytest.param("postgres", id="neon", marks=pytest.mark.live),
    ]
)
def store(request):
    """Yields each implementation in turn, isolated to a unique user id."""
    if request.param == "postgres":
        if not config.DATABASE_URL:
            pytest.skip("DATABASE_URL not configured")
        impl = _pg_store()
    else:
        impl = InMemoryStore()

    impl._test_user = f"test-{uuid.uuid4()}"          # noqa: SLF001
    yield impl
    impl.forget(impl._test_user)                       # noqa: SLF001


@pytest.fixture
def user(store):
    return store._test_user                            # noqa: SLF001


@pytest.fixture
def populated(store, user):
    for memory_type, content, agent in MEMORIES:
        store.remember(user, memory_type, content, payload={"t": memory_type},
                       source_agent=agent)
    return store


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def test_remember_returns_the_stored_record(store, user):
    out = store.remember(user, "episodic", "Something happened.",
                         payload={"k": 1}, source_agent="judge", importance=0.9)
    assert out["user_id"] == user
    assert out["memory_type"] == "episodic"
    assert out["content"] == "Something happened."
    assert out["payload"] == {"k": 1}
    assert out["source_agent"] == "judge"
    assert out["importance"] == 0.9
    assert out["id"]


@pytest.mark.parametrize("memory_type", MEMORY_TYPES)
def test_all_six_memory_types_are_writable(store, user, memory_type):
    out = store.remember(user, memory_type, f"a {memory_type} memory")
    assert out["memory_type"] == memory_type


def test_unknown_memory_type_is_rejected(store, user):
    with pytest.raises(ValueError, match="unknown memory_type"):
        store.remember(user, "telepathic", "nope")


# --------------------------------------------------------------------------- #
# Semantic recall
# --------------------------------------------------------------------------- #

def test_recall_ranks_the_relevant_memory_first(populated, user):
    hits = populated.recall(user, "emergency fund versus debt repayment", limit=4)
    assert hits[0]["memory_type"] == "episodic"
    assert hits[0]["similarity"] > hits[-1]["similarity"]


def test_recall_finds_a_behavioural_pattern(populated, user):
    hits = populated.recall(user, "dining overspending habit", limit=2)
    assert hits[0]["memory_type"] == "behavioral"


def test_recall_respects_the_limit(populated, user):
    assert len(populated.recall(user, "anything", limit=2)) == 2


def test_recall_can_be_scoped_to_types(populated, user):
    hits = populated.recall(user, "money", limit=10, memory_types=["goal", "simulation"])
    assert {h["memory_type"] for h in hits} <= {"goal", "simulation"}


def test_recall_on_an_unknown_user_is_empty(store):
    assert store.recall("nobody-at-all", "anything") == []


def test_recall_never_crosses_users(store, user):
    store.remember(user, "episodic", "mine")
    other = f"other-{uuid.uuid4()}"
    store.remember(other, "episodic", "theirs")
    try:
        hits = store.recall(user, "mine", limit=10)
        assert all(h["user_id"] == user for h in hits)
        assert all(h["content"] != "theirs" for h in hits)
    finally:
        store.forget(other)


def test_similarity_is_bounded(populated, user):
    for hit in populated.recall(user, "emergency fund", limit=4):
        assert -1.0 <= hit["similarity"] <= 1.0


# --------------------------------------------------------------------------- #
# Recency
# --------------------------------------------------------------------------- #

def test_recent_returns_newest_first(populated, user):
    kinds = [m["memory_type"] for m in populated.recent(user, limit=10)]
    assert kinds[0] == "simulation"          # written last
    assert len(kinds) == len(MEMORIES)


def test_recent_can_be_scoped_to_types(populated, user):
    out = populated.recent(user, memory_types=["goal"])
    assert [m["memory_type"] for m in out] == ["goal"]


def test_recent_respects_the_limit(populated, user):
    assert len(populated.recent(user, limit=2)) == 2


# --------------------------------------------------------------------------- #
# Forgetting
# --------------------------------------------------------------------------- #

def test_forget_removes_everything_for_a_user(populated, user):
    assert populated.forget(user) == len(MEMORIES)
    assert populated.recent(user) == []


def test_forget_can_target_one_type(populated, user):
    assert populated.forget(user, "goal") == 1
    assert "goal" not in {m["memory_type"] for m in populated.recent(user)}
    assert len(populated.recent(user)) == len(MEMORIES) - 1


def test_forget_on_an_empty_user_returns_zero(store):
    assert store.forget(f"ghost-{uuid.uuid4()}") == 0


# --------------------------------------------------------------------------- #
# Profile document
# --------------------------------------------------------------------------- #

def test_profile_round_trip(store, user):
    store.save_profile(user, {"name": "Test", "monthly_income": 90_000})
    assert store.load_profile(user) == {"name": "Test", "monthly_income": 90_000}


def test_profile_save_is_an_upsert(store, user):
    store.save_profile(user, {"v": 1})
    store.save_profile(user, {"v": 2})
    assert store.load_profile(user) == {"v": 2}


def test_missing_profile_is_none(store):
    assert store.load_profile(f"ghost-{uuid.uuid4()}") is None


# --------------------------------------------------------------------------- #
# Store selection
# --------------------------------------------------------------------------- #

def test_force_memory_returns_the_offline_store():
    assert isinstance(get_store(force_memory=True), InMemoryStore)


def test_set_store_overrides_the_default():
    sentinel = InMemoryStore()
    set_store(sentinel)
    assert get_store() is sentinel
