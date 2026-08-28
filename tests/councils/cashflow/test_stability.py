"""Cashflow Council -> Cashflow Stability Agent."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ml.src.councils.cashflow.income_projection import income_projection_node
from ml.src.councils.cashflow.stability import (
    cashflow_simulator,
    risk_engine,
    stability_node,
)
from ml.src.schemas.state import new_state

FIXED_TODAY = datetime(2026, 1, 15)


def _sim(balance=100_000, months=3, income=80_000, expense=50_000):
    return cashflow_simulator(
        current_balance=balance,
        income_fc=[income] * months,
        expense_fc_base=[expense] * months,
        expense_fc_p10=[expense * 0.9] * months,
        expense_fc_p90=[expense * 1.2] * months,
        months_ahead=months,
        today=FIXED_TODAY,
    )


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

def test_simulator_returns_one_row_per_month_indexed_by_label():
    df = _sim(months=6)
    assert len(df) == 6
    assert df.index.name == "month"


def test_month_labels_follow_the_injected_today():
    """`today` is a parameter, so labels are deterministic in tests."""
    df = _sim(months=3)
    assert list(df.index) == ["Feb 2026", "Mar 2026", "Apr 2026"]


def test_balance_compounds_month_over_month():
    df = _sim(balance=100_000, months=3, income=80_000, expense=50_000)
    assert list(df["bal_base"]) == [130_000, 160_000, 190_000]


def test_pessimistic_never_exceeds_base_never_exceeds_optimistic():
    df = _sim(months=6)
    assert (df["bal_pess"] <= df["bal_base"]).all()
    assert (df["bal_base"] <= df["bal_opt"]).all()


def test_deficit_drives_balance_negative():
    df = _sim(balance=10_000, months=3, income=30_000, expense=50_000)
    assert (df["bal_base"] < 0).any()


def test_expected_columns_are_present():
    df = _sim()
    assert {"income", "exp_base", "exp_opt", "exp_pess",
            "net_base", "bal_base", "bal_opt", "bal_pess"} <= set(df.columns)


# --------------------------------------------------------------------------- #
# Risk engine
# --------------------------------------------------------------------------- #

def _risk(df, balance=100_000, expenses=50_000, dependents=1):
    return risk_engine(
        sim_df=df, current_balance=balance, monthly_expenses_avg=expenses,
        dependents=dependents, goal_text="buy a house",
        external_factors_text="none stated",
    )


def test_healthy_projection_scores_low_with_no_flags():
    out = _risk(_sim(balance=500_000, income=120_000, expense=40_000, months=6))
    assert out["score"] < 25
    assert out["rating"].endswith("LOW")
    assert out["flags"] == []


def test_negative_balance_is_flagged_and_scored():
    out = _risk(_sim(balance=5_000, income=30_000, expense=60_000, months=6), balance=5_000)
    assert out["score"] >= 30
    assert any("Negative balance" in f for f in out["flags"])


def test_score_is_capped_at_100_and_rating_is_consistent():
    out = _risk(_sim(balance=0, income=1_000, expense=90_000, months=12), balance=0)
    assert 0 <= out["score"] <= 100
    assert out["rating"].split()[-1] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_income_volatility_defaults_to_the_simulated_income_column():
    """`income_values` was a notebook global; it must default, not raise."""
    out = _risk(_sim())
    assert "score" in out


def test_risk_contract_keys():
    out = _risk(_sim())
    assert set(out) == {
        "score", "rating", "flags", "savings_rate",
        "emergency_target", "min_projected_balance",
    }


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_uses_monte_carlo_when_expense_history_exists(history_profile):
    patch = stability_node(new_state(history_profile), months_ahead=6)
    assert set(patch) == {"stability_result", "simulation_result"}
    assert patch["simulation_result"]["engine"] == "monte_carlo"
    assert len(patch["stability_result"]["projection"]) == 6


def test_node_falls_back_to_flat_expenses_without_history(salaried_profile):
    patch = stability_node(new_state(salaried_profile), months_ahead=6)
    assert patch["simulation_result"]["engine"] == "flat"
    base = patch["stability_result"]["expense_forecast_base"]
    assert base == [salaried_profile.essential_expenses] * 6


def test_node_consumes_the_upstream_income_forecast(history_profile):
    """The two cashflow agents must compose: stability reuses projection's output."""
    state = new_state(history_profile)
    state.update(income_projection_node(state, months_ahead=6))
    expected = state["income_projection_result"]["forecast"]

    state.update(stability_node(state, months_ahead=6))
    assert state["stability_result"]["income_forecast"] == pytest.approx(expected)


def test_node_works_standalone_without_an_upstream_forecast(history_profile):
    patch = stability_node(new_state(history_profile), months_ahead=3)
    assert patch["stability_result"]["income_forecast"] == [74_000.0] * 3


def test_node_pads_a_short_upstream_forecast_to_the_horizon(history_profile):
    state = new_state(history_profile)
    state["income_projection_result"] = {"forecast": [70_000.0, 71_000.0]}
    patch = stability_node(state, months_ahead=6)
    assert len(patch["stability_result"]["income_forecast"]) == 6


def test_node_handles_the_zero_profile_without_error(zero_profile):
    patch = stability_node(new_state(zero_profile), months_ahead=3)
    risk = patch["stability_result"]["risk"]
    assert 0 <= risk["score"] <= 100
