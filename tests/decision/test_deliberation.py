"""
Decision Layer -> Multi-Agent Deliberation Engine + Master Judge.

Every test stubs `common.llm.chat`, so the council is exercised end-to-end with
no network and deterministic output.
"""

from __future__ import annotations

import pytest

import ml.src.common.llm as llm_module
from ml.src.decision.deliberation import (
    DEFAULT_PERSONAS,
    Persona,
    build_deliberation_graph,
    count_tokens,
    deliberate,
    make_advisor,
    make_critic,
    make_judge,
)
from ml.src.schemas.state import new_state


@pytest.fixture
def stub_chat(monkeypatch):
    """Deterministic LLM. Records every prompt it was given."""
    prompts: list[str] = []

    def _chat(prompt, **kwargs):
        prompts.append(prompt)
        return f"ADVICE#{len(prompts)}"

    monkeypatch.setattr(llm_module, "chat", _chat)
    return prompts


@pytest.fixture
def failing_chat(monkeypatch):
    def _chat(prompt, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_module, "chat", _chat)


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #

def test_count_tokens_is_zero_for_empty_text():
    assert count_tokens("") == 0


def test_count_tokens_is_positive_and_grows_with_length():
    short = count_tokens("one two three")
    long = count_tokens("one two three " * 50)
    assert short > 0
    assert long > short


def test_count_tokens_never_requires_a_gated_tokenizer(monkeypatch):
    """The Codestral tokenizer is gated; absent it we must still return an estimate."""
    import ml.src.decision.deliberation as mod

    monkeypatch.setattr(mod, "_tokenizer", lambda: None)
    assert count_tokens("a b c d e") >= 1


# --------------------------------------------------------------------------- #
# Graph topology
# --------------------------------------------------------------------------- #

def test_graph_has_the_notebook_s_nine_nodes():
    nodes = set(build_deliberation_graph().get_graph().nodes.keys())
    assert nodes == {
        "__start__", "__end__",
        "conservative", "growth", "value",
        "conservative_critic", "growth_critic", "value_critic",
        "judge",
    }


def test_personas_form_a_round_robin_critique_cycle():
    mapping = {p.key: p.critiques for p in DEFAULT_PERSONAS}
    assert mapping == {
        "conservative": "growth",
        "growth": "value",
        "value": "conservative",
    }


def test_graph_rejects_a_persona_that_critiques_an_unknown_key():
    bad = (Persona("a", "A", "brief", critiques="nobody"),)
    with pytest.raises(ValueError, match="not in the set"):
        build_deliberation_graph(bad)


def test_graph_rejects_an_empty_persona_set():
    with pytest.raises(ValueError, match="at least one persona"):
        build_deliberation_graph(())


def test_custom_persona_set_builds_its_own_graph():
    personas = (
        Persona("bull", "Bull", "argue for risk", critiques="bear"),
        Persona("bear", "Bear", "argue for caution", critiques="bull"),
    )
    nodes = set(build_deliberation_graph(personas).get_graph().nodes.keys())
    assert {"bull", "bear", "bull_critic", "bear_critic", "judge"} <= nodes


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #

def test_full_deliberation_accumulates_three_verdicts(stub_chat, salaried_profile):
    out = deliberate(new_state(salaried_profile, query="equity or FD?"))
    assert {v["agent"] for v in out["verdicts"]} == {"conservative", "growth", "value"}


def test_parallel_advisors_do_not_clobber_each_other(stub_chat, salaried_profile):
    """
    The three advisors fan out from START concurrently. Without `operator.add`
    reducers on `verdicts`, LangGraph raises InvalidUpdateError here.
    """
    out = deliberate(new_state(salaried_profile, query="q"))
    assert len(out["verdicts"]) == 3
    assert len(out["critiques"]) == 3


def test_each_critic_receives_the_advice_it_is_paired_with(stub_chat, salaried_profile):
    out = deliberate(new_state(salaried_profile, query="q"))
    stances = {c["agent"]: c["stance"] for c in out["critiques"]}
    assert stances == {
        "conservative": "critique_of_growth",
        "growth": "critique_of_value",
        "value": "critique_of_conservative",
    }


def test_judge_sees_every_verdict_and_critique(stub_chat, salaried_profile):
    deliberate(new_state(salaried_profile, query="q"))
    judge_prompt = stub_chat[-1]
    # The judge is asked for a bounded, plain-prose answer -- without a length
    # constraint the councils returned enough markdown to make the page 16,000
    # pixels tall.
    assert "Produce the final recommendation" in judge_prompt
    assert "At most 150 words" in judge_prompt
    assert "no markdown" in judge_prompt
    assert judge_prompt.count("ADVICE#") == 6      # 3 advisors + 3 critics


def test_final_decision_is_populated(stub_chat, salaried_profile):
    out = deliberate(new_state(salaried_profile, query="q"))
    assert out["final_decision"].startswith("ADVICE#")


def test_total_tokens_accumulates_across_all_seven_calls(stub_chat, salaried_profile):
    out = deliberate(new_state(salaried_profile, query="q"))
    per_agent = sum(v["tokens"] for v in out["verdicts"])
    per_critic = sum(c["tokens"] for c in out["critiques"])
    assert out["total_tokens"] > per_agent + per_critic     # judge adds its own
    assert len(stub_chat) == 7                              # 3 + 3 + 1


def test_query_reaches_every_advisor(stub_chat, salaried_profile):
    deliberate(new_state(salaried_profile, query="UNIQUE_MARKER_42"))
    advisor_prompts = stub_chat[:3]
    assert all("UNIQUE_MARKER_42" in p for p in advisor_prompts)


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #

def test_provider_failure_is_captured_not_raised(failing_chat, salaried_profile):
    out = deliberate(new_state(salaried_profile, query="q"))
    assert out["final_decision"] == ""
    assert len(out["errors"]) >= 3
    assert any("provider unavailable" in e for e in out["errors"])


def test_judge_reports_when_there_is_nothing_to_synthesise(salaried_profile):
    judge = make_judge()
    out = judge(new_state(salaried_profile, query="q"))
    assert out["final_decision"] == ""
    assert "no verdicts" in out["errors"][0]


def test_critic_reports_a_missing_target(stub_chat, salaried_profile):
    critic = make_critic(DEFAULT_PERSONAS[0])       # critiques "growth"
    out = critic(new_state(salaried_profile, query="q"))   # no verdicts present
    assert "no growth advice" in out["errors"][0]


# --------------------------------------------------------------------------- #
# Node factories in isolation
# --------------------------------------------------------------------------- #

def test_advisor_node_writes_one_verdict(stub_chat, salaried_profile):
    advisor = make_advisor(DEFAULT_PERSONAS[1])     # growth
    out = advisor(new_state(salaried_profile, query="q"))
    assert len(out["verdicts"]) == 1
    assert out["verdicts"][0]["agent"] == "growth"
    assert out["verdicts"][0]["stance"] == "Growth"


def test_node_names_are_stable_for_debugging():
    assert make_advisor(DEFAULT_PERSONAS[0]).__name__ == "conservative_advisor"
    assert make_critic(DEFAULT_PERSONAS[0]).__name__ == "conservative_critic"
    assert make_judge().__name__ == "judge"
