"""
Workflows -> the eight flows from the deck, plus the orchestrator.

The acceptance criterion for Day 5 is here: every workflow terminates in a
memory write, and a query routes to the right one without being told.
"""

from __future__ import annotations

import json

import pytest

import ml.src.common.llm as llm_module
from ml.src.memory.store import InMemoryStore, set_store
from ml.src.schemas.profile import Debt, Goal, UserProfile
from ml.src.schemas.state import RESULT_KEYS, new_state
from ml.src.workflows.base import AGENT_ORDER, agent_node, make_agent_runner, order_agents
from ml.src.workflows.catalogue import (
    INTENT_TO_WORKFLOW,
    WORKFLOWS,
    get_workflow,
    workflow_for_intent,
)
from ml.src.workflows.orchestrator import run, summarise_run


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No workflow test may hit a network."""
    monkeypatch.setattr(llm_module, "chat", lambda p, **k: "[council argument]")
    set_store(InMemoryStore())


@pytest.fixture
def profile() -> UserProfile:
    return UserProfile(
        user_id="wf", age=33, job_type="salaried", occupation="farmer",
        residence="rural", land_holding_ha=1.2, monthly_income=95_000,
        essential_expenses=48_000, existing_emergency_fund=40_000,
        retirement_corpus=300_000, dependents=2, has_health_insurance=False,
        annual_household_income=1_140_000,
        monthly_spend={"rent": 26_000, "dining": 8_000, "groceries": 8_000},
        debts=[Debt(name="Card", debt_type="credit_card", outstanding_amount=160_000,
                    interest_rate=42.0, minimum_due=8_000)],
        goals=[Goal(name="Home", target_amount=2_000_000, current_amount=250_000,
                    target_months=60)],
    )


# =========================================================================== #
# Agent ordering
# =========================================================================== #

def test_agent_order_covers_every_council_agent():
    assert set(AGENT_ORDER) == {k.removesuffix("_result") for k in RESULT_KEYS}


def test_agent_order_has_no_duplicates():
    assert len(AGENT_ORDER) == len(set(AGENT_ORDER))


@pytest.mark.parametrize(
    "earlier,later",
    [
        ("emergency_fund", "goal_allocation"),   # runway is reserved first
        ("emergency_fund", "asset_allocation"),  # allocation reads real runway
        ("debt_trap", "asset_allocation"),       # allocation reads leverage
        ("income_projection", "stability"),      # stability consumes the forecast
        ("goal_allocation", "retirement"),       # retirement uses leftover surplus
        ("bias_detection", "habit_formation"),   # habits target detected biases
        ("habit_formation", "nudge_strategy"),   # nudges reinforce the keystone
    ],
)
def test_dependencies_run_in_the_right_order(earlier, later):
    """
    Fanning these out concurrently would silently weaken every answer, because
    the downstream agent would see an empty state instead of its input.
    """
    assert AGENT_ORDER.index(earlier) < AGENT_ORDER.index(later)


def test_ordering_a_subset_preserves_dependency_order():
    assert order_agents(["retirement", "emergency_fund", "goal_allocation"]) == [
        "emergency_fund", "goal_allocation", "retirement",
    ]


def test_unknown_agents_are_dropped():
    assert order_agents(["emergency_fund", "not_an_agent"]) == ["emergency_fund"]


def test_every_named_agent_resolves_to_a_node():
    for name in AGENT_ORDER:
        assert callable(agent_node(name))


def test_a_failing_agent_does_not_abort_the_workflow(profile, monkeypatch):
    """A fraud lookup timing out must not cost the user their cashflow analysis."""
    import ml.src.councils.risk.emergency_fund as ef

    monkeypatch.setattr(ef, "emergency_fund_node",
                        lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    runner = make_agent_runner(["emergency_fund", "expense_optimizer"])
    patch = runner(new_state(profile))
    assert "expense_optimizer_result" in patch
    assert any("boom" in e for e in patch["errors"])


def test_later_agents_see_earlier_results(profile):
    runner = make_agent_runner(["emergency_fund", "goal_allocation"])
    patch = runner(new_state(profile))
    assert patch["goal_allocation_result"]["reserved_for_emergency"] > 0


# =========================================================================== #
# The catalogue
# =========================================================================== #

def test_all_eight_deck_workflows_plus_full_review_exist():
    assert len(WORKFLOWS) == 9
    assert {"credit_card", "salary_day", "income_simulation", "goal_planning",
            "life_event", "fraud", "benefits", "financial_resilience",
            "full_review"} == set(WORKFLOWS)


def test_every_workflow_declares_real_agents():
    known = set(AGENT_ORDER)
    for name, spec in WORKFLOWS.items():
        assert set(spec["agents"]) <= known, name
        assert spec["label"] and spec["description"]


def test_full_review_runs_every_agent():
    """
    Derived from AGENT_ORDER rather than the union of the other workflows --
    that union silently omitted `loan_advisor`.
    """
    assert set(WORKFLOWS["full_review"]["agents"]) == set(AGENT_ORDER)
    assert "loan_advisor" in WORKFLOWS["full_review"]["agents"]


def test_every_intent_maps_to_a_real_workflow():
    for intent, name in INTENT_TO_WORKFLOW.items():
        assert name in WORKFLOWS, intent


def test_cheap_questions_do_not_convene_the_councils():
    """Deliberation is expensive; a card question has a computable answer."""
    assert WORKFLOWS["credit_card"]["deliberate"] is False
    assert WORKFLOWS["benefits"]["deliberate"] is False
    assert WORKFLOWS["life_event"]["deliberate"] is True


def test_unknown_workflow_raises():
    with pytest.raises(ValueError, match="unknown workflow"):
        get_workflow("does_not_exist")


# =========================================================================== #
# Graph shape
# =========================================================================== #

@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_workflow_starts_with_recall_and_ends_with_memory(name):
    """Every flow diagram in the deck terminates in 'Update Memory'."""
    steps = get_workflow(name).workflow_steps
    assert steps[0] == "recall"
    assert steps[-1] == "remember"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_workflow_compiles_with_expected_nodes(name):
    graph = get_workflow(name)
    nodes = set(graph.get_graph().nodes)
    assert {"recall", "agents", "remember"} <= nodes


def test_declared_stages_appear_in_the_graph():
    steps = get_workflow("salary_day").workflow_steps
    assert "simulate" in steps and "deliberate" in steps and "optimise" in steps
    assert "deliberate" not in get_workflow("credit_card").workflow_steps


def test_workflows_are_cached():
    assert get_workflow("credit_card") is get_workflow("credit_card")


# =========================================================================== #
# Execution
# =========================================================================== #

@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_workflow_runs_and_writes_memory(name, profile):
    final = get_workflow(name).invoke(new_state(profile, query="test run"))
    assert final["memory_written"] is True, name


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_workflow_output_is_json_serialisable(name, profile):
    final = get_workflow(name).invoke(new_state(profile, query="test run"))
    json.dumps({k: v for k, v in final.items() if k != "profile"}, default=str)


def test_a_workflow_produces_its_declared_agents_results(profile):
    final = get_workflow("goal_planning").invoke(new_state(profile, query="goals"))
    for agent in WORKFLOWS["goal_planning"]["agents"]:
        assert final.get(f"{agent}_result"), agent


def test_deliberating_workflow_produces_council_verdicts(profile):
    final = get_workflow("life_event").invoke(new_state(profile, query="getting married"))
    assert len(final["verdicts"]) == 5
    assert final["final_decision"]


def test_deliberation_tokens_are_not_double_counted(profile):
    """
    `total_tokens` carries an `operator.add` reducer, so returning the sub-graph's
    accumulated total instead of the delta would count it twice.
    """
    final = get_workflow("life_event").invoke(new_state(profile, query="getting married"))
    council_tokens = sum(v["tokens"] for v in final["verdicts"])
    council_tokens += sum(c["tokens"] for c in final["critiques"])
    assert final["total_tokens"] >= council_tokens
    assert final["total_tokens"] < council_tokens * 2


# =========================================================================== #
# Orchestrator
# =========================================================================== #

@pytest.mark.parametrize(
    "query,workflow",
    [
        ("which credit card should I get?", "credit_card"),
        ("my salary just got credited", "salary_day"),
        ("how much will I have in 6 months?", "income_simulation"),
        ("I want to save 20 lakh for a house", "goal_planning"),
        ("we're getting married next year", "life_event"),
        ("is Doubler Capital a scam?", "fraud"),
        ("am I eligible for government schemes?", "benefits"),
        ("do I have enough emergency fund and insurance?", "financial_resilience"),
        ("give me a full financial review", "full_review"),
    ],
)
def test_queries_route_to_the_right_workflow(query, workflow, profile):
    result = run(query, profile, use_llm_router=False)
    assert result["workflow"] == workflow


def test_a_narrow_query_activates_few_agents(profile):
    assert run("which credit card should I get?", profile,
               use_llm_router=False)["agent_count"] == 1


def test_a_full_review_activates_every_agent(profile):
    result = run("give me a full financial review", profile, use_llm_router=False)
    assert result["agent_count"] == len(AGENT_ORDER)
    assert result["deliberated"] is True
    assert result["simulated"] is True


def test_orchestrator_reports_a_trace(profile):
    result = run("we're getting married next year", profile, use_llm_router=False)
    assert result["steps"][0] == "recall" and result["steps"][-1] == "remember"
    assert result["routing"]["intent"] == "life_event"
    assert result["elapsed_seconds"] >= 0


def test_orchestrator_passes_transactions_through(profile):
    result = run("do I have enough emergency fund?", profile,
                 use_llm_router=False, transactions=[])
    assert result["state"]["bias_detection_result"]["months_analysed"] == 0


def test_summary_is_json_safe_and_compact(profile):
    summary = summarise_run(run("give me a full financial review", profile,
                                use_llm_router=False))
    json.dumps(summary)
    assert "profile" not in summary
    assert summary["recommendations"]
    assert summary["council_verdicts"]


def test_memory_persists_across_two_runs(profile):
    run("give me a full financial review", profile, use_llm_router=False)
    second = run("how am I doing?", profile, use_llm_router=False)
    assert second["recalled"] > 0
