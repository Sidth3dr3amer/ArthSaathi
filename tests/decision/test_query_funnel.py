"""
Decision Layer -> Query Funnel.

Only `plan_query` / `explain` / `ask` call an LLM. Everything below exercises
the pure funnel stages, so the file runs offline.
"""

from __future__ import annotations

import pytest

from ml.src.decision.query_funnel import (
    _COMPARATORS,
    apply_record_filter,
    compute_aggregate,
    discover_schema,
    filter_by_domain,
    project_fields,
    run_funnel,
)

ROWS = [
    {"domain": "emergency_fund", "strategy_id": "S1", "months": 12, "amount": 600_000},
    {"domain": "emergency_fund", "strategy_id": "S2", "months": 6, "amount": 300_000},
    {"domain": "debt_plan", "strategy_id": "S3", "months": 3, "amount": 150_000},
]


# --------------------------------------------------------------------------- #
# Schema discovery
# --------------------------------------------------------------------------- #

def test_schema_lists_the_available_fields():
    schema = discover_schema(ROWS)
    assert "strategy_id" in schema
    assert "amount" in schema


def test_schema_of_no_records_is_empty():
    assert discover_schema([]) == {}


# --------------------------------------------------------------------------- #
# Domain filter — the "do not drop everything" safeguard
# --------------------------------------------------------------------------- #

def test_domain_filter_narrows_to_the_named_domain():
    out = filter_by_domain(ROWS, ["emergency_fund"])
    assert [r["strategy_id"] for r in out] == ["S1", "S2"]


def test_empty_domain_list_is_a_no_op():
    assert filter_by_domain(ROWS, []) == ROWS


def test_invalid_domains_fall_back_to_all_rows_rather_than_none():
    """
    A hallucinating planner must not silently empty the result set -- the
    notebook added this guard deliberately.
    """
    assert filter_by_domain(ROWS, ["not_a_domain"]) == ROWS


def test_partially_valid_domains_keep_only_the_valid_ones():
    out = filter_by_domain(ROWS, ["debt_plan", "nonsense"])
    assert [r["strategy_id"] for r in out] == ["S3"]


def test_rows_without_a_domain_key_do_not_crash():
    rows = [{"strategy_id": "X", "amount": 1}]
    assert filter_by_domain(rows, ["emergency_fund"]) == rows


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def test_projection_keeps_the_requested_fields():
    out = project_fields(ROWS, ["strategy_id"])
    assert all("strategy_id" in r for r in out)
    assert all("months" not in r for r in out)


def test_empty_field_list_returns_rows_unchanged():
    assert project_fields(ROWS, []) == ROWS


# --------------------------------------------------------------------------- #
# Record filter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "expr,expected",
    [
        ("months greater than 6", ["S1"]),
        ("months less than 6", ["S3"]),
        ("months at least 6", ["S1", "S2"]),
        ("months at most 6", ["S2", "S3"]),
        ("months equals 12", ["S1"]),
    ],
)
def test_comparison_predicates(expr, expected):
    out = apply_record_filter(ROWS, expr)
    assert [r["strategy_id"] for r in out] == expected


def test_all_records_filter_is_a_no_op():
    assert apply_record_filter(ROWS, "all records") == ROWS


def test_unparseable_filter_returns_rows_unchanged():
    assert apply_record_filter(ROWS, "something the planner made up") == ROWS


def test_comparator_vocabulary_is_populated():
    assert {"greater than", "less than", "at least", "at most"} <= set(_COMPARATORS)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "computation,expected",
    [("sum", 1_050_000.0), ("average", 350_000.0), ("count", 3),
     ("min", 150_000.0), ("max", 600_000.0)],
)
def test_aggregations(computation, expected):
    """Supported vocabulary is sum / average / count / max / min."""
    out = compute_aggregate(ROWS, computation, "amount")
    assert out["result"] == pytest.approx(expected)
    assert out["n"] == 3


def test_unknown_computation_yields_a_null_result_not_an_error():
    out = compute_aggregate(ROWS, "compare", "amount")
    assert out["result"] is None
    assert out["n"] == 3


@pytest.mark.parametrize("computation,field", [("none", "amount"), ("", "amount"), ("sum", "")])
def test_no_computation_requested_returns_none_entirely(computation, field):
    assert compute_aggregate(ROWS, computation, field) is None


def test_aggregate_over_a_missing_field_is_safe():
    out = compute_aggregate(ROWS, "sum", "not_a_field")
    assert out["n"] == 0
    assert out["result"] is None


def test_aggregate_over_no_rows_is_safe():
    out = compute_aggregate([], "sum", "amount")
    assert out["n"] == 0


# --------------------------------------------------------------------------- #
# Whole funnel
# --------------------------------------------------------------------------- #

def test_funnel_runs_all_four_stages_in_order():
    plan = {
        "is_specific": True,
        "domains": ["emergency_fund"],
        "fields": ["strategy_id", "amount"],
        "record_filter": "months greater than 6",
        "computation": "sum",
        "computation_field": "amount",
    }
    out = run_funnel(ROWS, plan)
    assert out["skipped"] is False
    assert out["funnel_trace"] == {
        "after_domain_filter": 2,
        "after_record_filter": 1,
        "after_field_projection": 1,
    }
    assert out["aggregate"]["result"] == pytest.approx(600_000.0)
    assert [r["strategy_id"] for r in out["narrowed_rows"]] == ["S1"]


def test_vague_question_short_circuits_the_funnel():
    """`is_specific: False` means the planner could not ground the question."""
    out = run_funnel(ROWS, {"is_specific": False})
    assert out["skipped"] is True
    assert out["narrowed_rows"] == []
    assert out["aggregate"] is None


def test_funnel_tolerates_a_plan_with_only_is_specific():
    out = run_funnel(ROWS, {"is_specific": True})
    assert out["skipped"] is False
    assert out["funnel_trace"]["after_domain_filter"] == len(ROWS)


def test_funnel_does_not_mutate_the_input_rows():
    before = [dict(r) for r in ROWS]
    run_funnel(ROWS, {"is_specific": True, "domains": ["emergency_fund"],
                      "fields": ["strategy_id"]})
    assert ROWS == before
