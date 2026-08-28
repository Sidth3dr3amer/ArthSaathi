"""
Profile Agent -> the six slide-8 components plus RAG.

The whole pipeline is exercised with a stubbed LLM, so the file runs offline.
"""

from __future__ import annotations

import json

import pytest

import ml.src.common.llm as llm_module
from ml.src.memory.store import InMemoryStore
from ml.src.profile_agent import memory_creator, question_gen, rag, updater
from ml.src.profile_agent.extractor import EXTRACTABLE, extract_information, parse_extraction
from ml.src.profile_agent.input_processor import (
    detect_language,
    extract_amounts,
    normalise_text,
    process_input,
)
from ml.src.profile_agent.response_gen import (
    answer_with_context,
    compose_draft,
    generate_response,
    run_profile_agent,
)
from ml.src.schemas.profile import Debt, UserProfile


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def blank():
    return UserProfile(user_id="rahul", name="Rahul")


@pytest.fixture
def stub_extractor(monkeypatch):
    """Returns a setter so a test can script successive extractor replies."""
    replies: list[str] = []

    def _chat(prompt, system=None, **kwargs):
        if system and "extract" in system.lower():
            return replies.pop(0) if replies else "{}"
        return "[natural reply]"

    monkeypatch.setattr(llm_module, "chat", _chat)
    return replies


# =========================================================================== #
# Input Processor
# =========================================================================== #

@pytest.mark.parametrize(
    "text,expected",
    [
        ("15k", 15_000), ("2.5 lakh", 250_000), ("1.2 crore", 12_000_000),
        ("35 thousand", 35_000), ("1,20,000", 120_000), ("120000", 120_000),
    ],
)
def test_indian_number_forms_are_parsed(text, expected):
    amounts = extract_amounts(normalise_text(text))
    assert amounts and amounts[0]["value"] == expected


def test_devanagari_amounts_are_parsed():
    """Voice input arrives in the user's own script."""
    amounts = extract_amounts(normalise_text("मेरी income 15 हज़ार है"))
    assert amounts[0]["value"] == 15_000


def test_currency_noise_is_stripped():
    assert "Rs" not in normalise_text("Rs. 1,20,000/-")
    assert "₹" not in normalise_text("₹5000")


def test_surface_form_is_kept_for_echoing_back():
    amounts = extract_amounts(normalise_text("about 2.5 lakh saved"))
    assert amounts[0]["surface"] == "2.5 lakh"


@pytest.mark.parametrize(
    "text,lang",
    [("I earn 50000", "en"), ("मेरी income 15 हज़ार है", "hi"),
     ("ನನ್ನ ಸಂಬಳ", "kn"), ("என் வருமானம்", "ta")],
)
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_empty_input_is_flagged():
    out = process_input("")
    assert out["is_empty"] is True
    assert out["amounts"] == []


def test_amounts_are_ranked_largest_first():
    out = process_input("I earn 50000 and spend 20000")
    assert [a["value"] for a in out["amounts"]] == [50_000, 20_000]


# =========================================================================== #
# Information Extractor
# =========================================================================== #

def test_json_is_parsed_out_of_surrounding_prose():
    assert parse_extraction('Sure! {"age": 30} hope that helps') == {"age": 30}


def test_unknown_keys_are_discarded():
    """A hallucinated field must never enter the profile."""
    out = parse_extraction('{"age": 30, "net_worth": 5000000, "secret": "x"}')
    assert out == {"age": 30}


def test_values_are_coerced_to_the_declared_type():
    out = parse_extraction('{"age": "34", "monthly_income": "50000", "has_health_insurance": "yes"}')
    assert out == {"age": 34, "monthly_income": 50_000.0, "has_health_insurance": True}


def test_invalid_enum_values_are_dropped():
    assert parse_extraction('{"job_type": "astronaut"}') == {}
    assert parse_extraction('{"job_type": "salaried"}') == {"job_type": "salaried"}


def test_malformed_json_yields_nothing():
    assert parse_extraction("not json at all") == {}
    assert parse_extraction('{"broken": ') == {}


def test_extraction_rejects_values_failing_profile_validation(stub_extractor):
    stub_extractor.append('{"age": 900, "monthly_income": 50000}')
    out = extract_information(process_input("I am 900 years old and earn 50000"))
    assert "age" not in out["fields"]
    assert out["fields"]["monthly_income"] == 50_000
    assert any(r["field"] == "age" for r in out["rejected"])


def test_regex_corroborated_values_score_higher_confidence(stub_extractor):
    stub_extractor.append('{"monthly_income": 35000, "dependents": 3}')
    out = extract_information(process_input("I earn 35 thousand, 3 dependants"))
    assert out["confidence"]["monthly_income"] > out["confidence"]["dependents"]


def test_extractor_survives_a_provider_outage(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(llm_module, "chat", boom)
    out = extract_information(process_input("I earn 50000"))
    assert out["method"] == "unavailable"
    assert out["fields"] == {}


def test_empty_input_short_circuits_the_extractor():
    assert extract_information(process_input(""))["method"] == "empty_input"


def test_every_extractable_field_exists_on_the_profile():
    for field in EXTRACTABLE:
        assert hasattr(UserProfile(), field), field


# =========================================================================== #
# Profile Updater
# =========================================================================== #

def test_unset_fields_are_filled(blank):
    merged = updater.merge_fields(blank, {"monthly_income": 35_000}, {"monthly_income": 0.7})
    assert merged["profile"].monthly_income == 35_000
    assert merged["changed"] is True


def test_a_low_confidence_value_never_overwrites_an_existing_one():
    profile = UserProfile(user_id="x", monthly_income=50_000)
    merged = updater.merge_fields(profile, {"monthly_income": 60_000}, {"monthly_income": 0.5})
    assert merged["profile"].monthly_income == 50_000
    assert merged["needs_confirmation"][0]["field"] == "monthly_income"


def test_a_suspicious_jump_is_held_for_confirmation():
    """A misheard income poisons every downstream council, so it is queried."""
    profile = UserProfile(user_id="x", monthly_income=35_000)
    merged = updater.merge_fields(profile, {"monthly_income": 3_500_000},
                                  {"monthly_income": 0.95})
    assert merged["profile"].monthly_income == 35_000
    assert "3x" in merged["needs_confirmation"][0]["reason"]


def test_a_plausible_change_is_applied():
    profile = UserProfile(user_id="x", monthly_income=35_000)
    merged = updater.merge_fields(profile, {"monthly_income": 45_000},
                                  {"monthly_income": 0.95})
    assert merged["profile"].monthly_income == 45_000


def test_unchanged_values_are_skipped():
    profile = UserProfile(user_id="x", monthly_income=35_000)
    merged = updater.merge_fields(profile, {"monthly_income": 35_000}, {"monthly_income": 0.9})
    assert merged["changed"] is False
    assert merged["skipped"][0]["reason"] == "unchanged"


def test_merging_never_mutates_the_input(blank):
    before = blank.model_dump()
    updater.merge_fields(blank, {"monthly_income": 99_000}, {"monthly_income": 0.9})
    assert blank.model_dump() == before


def test_profile_round_trips_through_the_store(store, blank):
    profile = blank.model_copy(update={"monthly_income": 42_000})
    assert updater.save_profile(profile, store)["saved"] is True
    assert updater.load_profile("rahul", store).monthly_income == 42_000


def test_loading_an_unknown_user_returns_none(store):
    assert updater.load_profile("nobody", store) is None


def test_a_store_outage_does_not_raise(blank):
    class Broken(InMemoryStore):
        def save_profile(self, *a, **k):
            raise RuntimeError("neon down")

    out = updater.save_profile(blank, Broken())
    assert out["saved"] is False and "neon down" in out["error"]


# =========================================================================== #
# Memory Creator
# =========================================================================== #

def test_learned_facts_become_semantic_memories(store):
    out = memory_creator.create_memories(
        "rahul", [{"field": "monthly_income", "from": 0, "to": 35_000}],
        user_message="I earn 35 thousand", store=store,
    )
    assert out["memories_written"] == 2         # one semantic + one episodic
    kinds = {m["memory_type"] for m in store.recent("rahul")}
    assert kinds == {"semantic", "episodic"}


def test_semantic_memory_is_phrased_readably(store):
    memory_creator.create_memories(
        "rahul", [{"field": "monthly_income", "from": 0, "to": 35_000}], store=store
    )
    semantic = store.recent("rahul", memory_types=["semantic"])[0]
    assert "monthly income is Rs 35,000" in semantic["content"]


def test_memory_creation_survives_a_broken_store():
    class Broken(InMemoryStore):
        def remember(self, *a, **k):
            raise RuntimeError("write failed")

    out = memory_creator.create_memories(
        "rahul", [{"field": "age", "from": 0, "to": 30}], store=Broken()
    )
    assert out["memories_written"] == 0
    assert out["errors"]


def test_phrasing_falls_back_for_unknown_fields():
    assert "something odd" in memory_creator.phrase("something_odd", 5)


# =========================================================================== #
# Question Generator
# =========================================================================== #

def test_the_most_valuable_question_is_asked_first(blank):
    assert question_gen.generate_question(blank)["field"] == "monthly_income"


def test_answered_fields_are_not_re_asked():
    profile = UserProfile(user_id="x", monthly_income=50_000)
    assert question_gen.generate_question(profile)["field"] != "monthly_income"


def test_can_advise_requires_the_essentials(blank):
    assert question_gen.completeness(blank)["can_advise"] is False
    ready = blank.model_copy(update={"monthly_income": 50_000, "essential_expenses": 25_000})
    assert question_gen.completeness(ready)["can_advise"] is True


def test_questioning_stops_once_only_low_value_fields_remain():
    profile = UserProfile(
        user_id="x", monthly_income=50_000, essential_expenses=25_000,
        dependents=2, age=35, job_type="salaried", existing_emergency_fund=100_000,
        occupation="engineer",
    )
    question = question_gen.generate_question(profile)
    assert question is None or question["weight"] >= 5


def test_progress_matches_the_onboarding_screen(blank):
    plan = question_gen.question_plan(blank, limit=7)
    assert len(plan["queue"]) == 7
    assert plan["completeness"]["total"] == len(question_gen.QUESTION_BANK)
    assert plan["next_question"]["progress"]["percent"] >= 0


def test_blocked_councils_are_named_when_essentials_are_missing(blank):
    assert question_gen.question_plan(blank)["blocked_councils"]


def test_every_question_explains_why_it_is_asked():
    for spec in question_gen.QUESTION_BANK.values():
        assert spec["question"].endswith("?")
        assert len(spec["why"]) > 20
        assert spec["unlocks"]


# =========================================================================== #
# RAG
# =========================================================================== #

def test_retrieval_returns_profile_and_memories(store, blank):
    store.save_profile("rahul", blank.model_dump(mode="json"))
    store.remember("rahul", "semantic", "User monthly income is Rs 35,000.")
    out = rag.retrieve("rahul", "what is my income?", store=store)
    assert out["profile"] is not None
    assert out["memories"]


def test_weak_matches_are_discarded(store):
    for i in range(5):
        store.remember("rahul", "episodic", f"completely unrelated content {i}")
    out = rag.retrieve("rahul", "quantum chromodynamics", store=store)
    assert out["discarded_weak"] >= 0
    assert all(m["similarity"] >= rag.SIMILARITY_FLOOR for m in out["memories"])


def test_context_stays_within_budget():
    """
    Budget trimming is tested directly rather than through `retrieve`, because
    the similarity floor would otherwise discard the filler first and leave
    nothing for the budget to trim.
    """
    retrieved = {
        "memories": [
            {"memory_type": "semantic", "content": f"Fact {i} " + "x" * 200,
             "similarity": 0.9 - i * 0.01}
            for i in range(20)
        ]
    }
    profile = UserProfile(user_id="rahul", monthly_income=50_000)
    built = rag.build_context(profile, retrieved, budget=500)

    assert built["memories_included"] < 20
    assert built["memories_dropped"] > 0
    assert built["characters"] < 500 + len(built["profile_summary"]) + 100


def test_context_keeps_the_strongest_matches_when_trimming():
    retrieved = {
        "memories": [
            {"memory_type": "semantic", "content": "STRONGEST " + "x" * 100, "similarity": 0.9},
            {"memory_type": "semantic", "content": "WEAKEST " + "y" * 100, "similarity": 0.1},
        ]
    }
    built = rag.build_context(UserProfile(user_id="r"), retrieved, budget=120)
    assert "STRONGEST" in built["context"]
    assert "WEAKEST" not in built["context"]


def test_profile_summary_only_states_what_is_known():
    summary = rag.profile_summary(UserProfile(user_id="x"))
    assert "Monthly income" not in summary
    assert "Health insurance" in summary


def test_profile_summary_includes_debts_and_goals():
    profile = UserProfile(
        user_id="x", monthly_income=90_000, essential_expenses=40_000,
        debts=[Debt(name="Card", debt_type="credit_card",
                    outstanding_amount=100_000, interest_rate=42.0)],
    )
    summary = rag.profile_summary(profile)
    assert "Debts: 1" in summary and "42%" in summary


def test_retrieval_survives_a_store_outage(blank):
    class Broken(InMemoryStore):
        def recall(self, *a, **k):
            raise RuntimeError("neon down")

    out = rag.retrieve("rahul", "anything", store=Broken())
    assert out["memories"] == []
    assert out["errors"]


# =========================================================================== #
# Response Generator + full pipeline
# =========================================================================== #

def test_draft_confirms_what_was_recorded():
    draft = compose_draft(
        {"applied": [{"field": "monthly_income", "from": 0, "to": 35_000}],
         "needs_confirmation": []},
        None, {"is_empty": False},
    )
    assert "recorded" in draft and "35,000" in draft


def test_draft_surfaces_held_back_changes_as_a_question():
    draft = compose_draft(
        {"applied": [], "needs_confirmation": [
            {"field": "monthly_income", "current": 35_000, "proposed": 3_500_000,
             "reason": "changes by more than 3x"}]},
        None, {"is_empty": False},
    )
    assert "Which is right?" in draft


def test_response_falls_back_to_the_draft_on_provider_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(llm_module, "chat", boom)
    out = generate_response("Noted your income.")
    assert out["response"] == "Noted your income."
    assert out["method"] == "fallback_draft"


def test_full_pipeline_builds_a_profile_over_turns(stub_extractor, store, blank):
    stub_extractor.extend([
        '{"monthly_income": 35000, "occupation": "farmer", "dependents": 3}',
        '{"essential_expenses": 22000}',
    ])
    first = run_profile_agent("I'm a farmer earning 35 thousand, 3 dependants",
                              blank, store=store)
    assert first["profile"].monthly_income == 35_000
    assert first["completeness"]["can_advise"] is False

    second = run_profile_agent("I spend about 22000 on essentials",
                               first["profile"], store=store)
    assert second["profile"].essential_expenses == 22_000
    assert second["completeness"]["can_advise"] is True


def test_full_pipeline_reports_every_stage(stub_extractor, store, blank):
    stub_extractor.append('{"monthly_income": 35000}')
    out = run_profile_agent("I earn 35 thousand", blank, store=store)
    assert set(out["stages"]) == {
        "input_processor", "extractor", "updater",
        "memory_creator", "question_generator", "response_generator",
    }
    json.dumps({k: v for k, v in out.items() if k != "profile"}, default=str)


def test_pipeline_holds_back_a_suspicious_correction(stub_extractor, store):
    stub_extractor.append('{"monthly_income": 3500000}')
    profile = UserProfile(user_id="rahul", monthly_income=35_000)
    out = run_profile_agent("my income is 35 lakh", profile, store=store)
    assert out["profile"].monthly_income == 35_000
    assert out["needs_confirmation"]


def test_rag_answer_uses_only_retrieved_context(monkeypatch, store):
    seen = {}
    monkeypatch.setattr(llm_module, "chat",
                        lambda p, **k: seen.update({"prompt": p}) or "answer")
    store.remember("rahul", "semantic", "User monthly income is Rs 35,000.")
    profile = UserProfile(user_id="rahul", monthly_income=35_000)
    out = answer_with_context("what do I earn?", profile, store=store)
    assert "WHAT WE KNOW ABOUT THIS USER" in seen["prompt"]
    assert out["answer"] == "answer"


def test_rag_answer_degrades_without_a_provider(monkeypatch, store):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(llm_module, "chat", boom)
    profile = UserProfile(user_id="rahul", monthly_income=35_000)
    out = answer_with_context("what do I earn?", profile, store=store)
    assert "on record" in out["answer"]
    assert out["method"].startswith("fallback")
