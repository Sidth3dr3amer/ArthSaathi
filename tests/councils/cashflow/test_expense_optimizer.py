"""Cashflow Council -> Expense Optimizer Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.cashflow.expense_optimizer import (
    BENCHMARKS,
    DEFAULT_BENCHMARK,
    expense_optimizer_advisor,
    expense_optimizer_node,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state

SPEND = {
    "rent": 30_000, "dining": 9_000, "groceries": 9_000,
    "entertainment": 6_000, "subscriptions": 3_000, "fuel": 4_000,
}


def advise(income=90_000, spend=None):
    return expense_optimizer_advisor(income, spend if spend is not None else dict(SPEND))


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #

def test_categories_within_benchmark_are_not_flagged():
    out = advise(spend={"groceries": 5_000})       # 5.6% vs 12% ceiling
    assert out["overspending"] == []
    assert out["potential_monthly_savings"] == 0


def test_category_over_benchmark_is_flagged_with_its_overshoot():
    out = advise(spend={"dining": 9_000})           # 10% vs 6% ceiling
    dining = out["categories"][0]
    assert dining["over_benchmark"] is True
    assert dining["overshoot"] == pytest.approx(9_000 - 90_000 * 0.06)


def test_unknown_category_uses_the_default_benchmark():
    out = advise(spend={"crypto_punts": 20_000})
    cat = out["categories"][0]
    assert cat["benchmark_share"] == DEFAULT_BENCHMARK[0]


def test_every_category_is_reported_not_just_the_overspends():
    out = advise()
    assert len(out["categories"]) == len(SPEND)


# --------------------------------------------------------------------------- #
# Recoverability — the "achievable, not theoretical" rule
# --------------------------------------------------------------------------- #

def test_recoverable_is_a_fraction_of_overshoot_not_all_of_it():
    out = advise(spend={"dining": 9_000})
    dining = out["categories"][0]
    assert 0 < dining["recoverable"] < dining["overshoot"]


def test_rent_overshoot_is_barely_recoverable_but_dining_mostly_is():
    """Structural costs cannot be cut this month; discretionary ones can."""
    rent_share = BENCHMARKS["rent"][1]
    dining_share = BENCHMARKS["dining"][1]
    assert rent_share < 0.2
    assert dining_share > 0.5


def test_categories_are_ranked_by_recoverable_amount():
    out = advise()
    recoverables = [c["recoverable"] for c in out["categories"]]
    assert recoverables == sorted(recoverables, reverse=True)


def test_headline_savings_is_the_sum_of_recoverables():
    out = advise()
    assert out["potential_monthly_savings"] == pytest.approx(
        sum(c["recoverable"] for c in out["categories"]), abs=0.02
    )
    assert out["potential_annual_savings"] == pytest.approx(
        out["potential_monthly_savings"] * 12, abs=0.5
    )


# --------------------------------------------------------------------------- #
# Status bands
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "spend_total,expected",
    [
        (100_000, "Spending exceeds income"),
        (85_000, "Critically tight"),
        (70_000, "Tight"),
        (55_000, "Comfortable"),
        (30_000, "Highly efficient"),
    ],
)
def test_status_bands(spend_total, expected):
    out = advise(spend={"others": spend_total})
    assert out["status"] == expected


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #

def test_zero_income_returns_insufficient_data_not_a_crash():
    out = expense_optimizer_advisor(0, {"dining": 5_000})
    assert out["status"] == "Insufficient data"
    assert out["potential_monthly_savings"] == 0


def test_no_spend_data_returns_insufficient_data():
    out = expense_optimizer_advisor(90_000, {})
    assert out["status"] == "Insufficient data"
    assert out["categories"] == []


def test_a_frugal_user_is_told_to_change_nothing():
    out = advise(income=200_000, spend={"groceries": 6_000, "dining": 3_000})
    assert out["overspending"] == []
    assert out["recommendations"] == []


def test_recommendations_are_capped_at_five():
    spend = {f"cat_{i}": 20_000 for i in range(10)}
    assert len(advise(spend=spend)["recommendations"]) == 5


def test_income_shares_are_fractions():
    out = advise()
    assert all(0 <= c["income_share"] <= 5 for c in out["categories"])


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(salaried_profile):
    patch = expense_optimizer_node(new_state(salaried_profile))
    assert set(patch) == {"expense_optimizer_result"}
    json.dumps(patch)


def test_node_falls_back_to_an_aggregate_bucket_without_a_breakdown():
    """A user with expenses but no categories still gets a spend ratio."""
    profile = UserProfile(user_id="agg", monthly_income=80_000, essential_expenses=50_000)
    result = expense_optimizer_node(new_state(profile))["expense_optimizer_result"]
    assert result["status"] != "Insufficient data"
    assert result["total_spend"] == 50_000


def test_node_survives_the_zero_profile(zero_profile):
    result = expense_optimizer_node(new_state(zero_profile))["expense_optimizer_result"]
    assert result["status"] == "Insufficient data"
