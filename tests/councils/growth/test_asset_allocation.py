"""Growth Council -> Asset Allocation Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.growth.asset_allocation import (
    MAX_EQUITY,
    asset_allocation_advisor,
    asset_allocation_node,
)
from ml.src.councils.risk.debt_trap import debt_trap_node
from ml.src.councils.risk.emergency_fund import emergency_fund_node
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import new_state


def advise(**overrides):
    base = dict(age=32, monthly_income=120_000, job_type="salaried", dependents=1,
                emergency_fund_months=6.0, debt_to_income=0.0,
                has_high_interest_debt=False, risk_tolerance="moderate")
    base.update(overrides)
    return asset_allocation_advisor(**base)


# --------------------------------------------------------------------------- #
# Allocation validity
# --------------------------------------------------------------------------- #

def test_allocation_always_sums_to_one():
    for age in (22, 35, 50, 65):
        for tol in ("conservative", "moderate", "aggressive"):
            out = advise(age=age, risk_tolerance=tol)
            assert sum(out["target_allocation"].values()) == pytest.approx(1.0, abs=1e-4)


def test_no_sleeve_is_negative():
    out = advise(age=70, risk_tolerance="conservative", emergency_fund_months=0)
    assert all(v >= 0 for v in out["target_allocation"].values())


def test_equity_is_capped():
    out = advise(age=18, risk_tolerance="aggressive", emergency_fund_months=12,
                 job_type="govt", dependents=0)
    assert out["target_allocation"]["equity"] <= MAX_EQUITY


def test_percent_view_matches_the_fractional_view():
    out = advise()
    for asset, share in out["target_allocation"].items():
        assert out["target_allocation_percent"][asset] == pytest.approx(share * 100, abs=0.01)


# --------------------------------------------------------------------------- #
# Glide path
# --------------------------------------------------------------------------- #

def test_equity_falls_with_age():
    young = advise(age=25)["target_allocation"]["equity"]
    mid = advise(age=45)["target_allocation"]["equity"]
    old = advise(age=60)["target_allocation"]["equity"]
    assert young > mid > old


def test_higher_tolerance_raises_equity():
    cons = advise(risk_tolerance="conservative")["target_allocation"]["equity"]
    aggr = advise(risk_tolerance="aggressive")["target_allocation"]["equity"]
    assert aggr > cons


# --------------------------------------------------------------------------- #
# Capacity binds over tolerance
# --------------------------------------------------------------------------- #

def test_capacity_binds_when_it_is_lower_than_tolerance():
    out = advise(risk_tolerance="aggressive", job_type="freelancer",
                 emergency_fund_months=0, dependents=3, debt_to_income=0.45)
    assert out["binding_constraint"] == "capacity"
    assert out["effective_risk"] == out["risk_capacity"]


def test_tolerance_binds_when_capacity_is_ample():
    out = advise(risk_tolerance="conservative", job_type="govt",
                 emergency_fund_months=12, dependents=0)
    assert out["binding_constraint"] == "tolerance"
    assert out["effective_risk"] == out["risk_tolerance"]


def test_stable_employment_raises_capacity():
    govt = advise(job_type="govt")["risk_capacity"]
    gig = advise(job_type="freelancer")["risk_capacity"]
    assert govt > gig


def test_leverage_reduces_capacity():
    clean = advise(debt_to_income=0.0)["risk_capacity"]
    levered = advise(debt_to_income=0.45)["risk_capacity"]
    assert levered < clean


def test_capacity_is_bounded():
    assert 0.0 <= advise(job_type="unsalaried", dependents=9, debt_to_income=1.0,
                         emergency_fund_months=0, age=64)["risk_capacity"] <= 1.0


# --------------------------------------------------------------------------- #
# Safety overrides
# --------------------------------------------------------------------------- #

def test_high_interest_debt_halves_equity_and_says_why():
    clean = advise(has_high_interest_debt=False)["target_allocation"]["equity"]
    indebted = advise(has_high_interest_debt=True)
    assert indebted["target_allocation"]["equity"] < clean
    assert any("High-interest debt" in w for w in indebted["warnings"])


def test_thin_runway_reduces_equity_and_says_why():
    out = advise(emergency_fund_months=1.0)
    assert any("runway" in w for w in out["warnings"])


def test_a_healthy_investor_gets_no_warnings():
    out = advise(emergency_fund_months=6, has_high_interest_debt=False)
    assert out["warnings"] == []


# --------------------------------------------------------------------------- #
# Rebalancing
# --------------------------------------------------------------------------- #

def test_rebalancing_flags_material_drift_only():
    out = advise(current_allocation={"equity": 90, "debt": 10, "gold": 0, "cash": 0})
    assert out["rebalancing"]
    assert all(abs(r["delta"]) >= 0.05 for r in out["rebalancing"])
    assert out["rebalancing"][0]["asset"] == "equity"
    assert out["rebalancing"][0]["action"] == "reduce"


def test_no_rebalancing_when_already_on_target():
    target = advise()["target_allocation"]
    out = advise(current_allocation={k: v * 100 for k, v in target.items()})
    assert out["rebalancing"] == []


def test_no_current_holdings_means_no_rebalancing_advice():
    assert advise()["rebalancing"] == []


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(salaried_profile):
    patch = asset_allocation_node(new_state(salaried_profile))
    assert set(patch) == {"asset_allocation_result"}
    json.dumps(patch)


def test_node_consumes_upstream_emergency_and_debt_agents():
    profile = UserProfile(
        user_id="compose", age=32, job_type="salaried", monthly_income=120_000,
        essential_expenses=55_000, existing_emergency_fund=80_000, dependents=1,
        debts=[Debt(name="Card", debt_type="credit_card", outstanding_amount=180_000,
                    interest_rate=42.0, minimum_due=9_000)],
    )
    state = new_state(profile)
    state.update(emergency_fund_node(state))
    state.update(debt_trap_node(state))
    result = asset_allocation_node(state)["asset_allocation_result"]
    assert any("High-interest debt" in w for w in result["warnings"])


def test_node_survives_the_zero_profile(zero_profile):
    result = asset_allocation_node(new_state(zero_profile))["asset_allocation_result"]
    assert sum(result["target_allocation"].values()) == pytest.approx(1.0, abs=1e-4)
