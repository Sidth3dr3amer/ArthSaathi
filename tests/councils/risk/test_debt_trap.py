"""
Risk Council -> Debt Trap Agent.

Golden values pinned from a live run of the migrated module against the same
inputs the notebook used, so refactors cannot silently change a payment plan.
"""

from __future__ import annotations

import pytest

from ml.src.councils.risk.debt_trap import (
    DEFAULT_RATES,
    EMI_DEBTS,
    allocate_payments,
    calculate_emi_arrears,
    calculate_loss_if_not_paid,
    calculate_mandatory_payments,
    debt_trap_node,
    get_debt_bucket,
    get_effective_rate,
    rank_debts,
)
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import new_state

CARD = {
    "name": "HDFC Card", "debt_type": "credit_card", "outstanding_amount": 140_000,
    "interest_rate": 42.0, "minimum_due": 7_000, "emi": 0, "overdue_cycles": 0,
}
EDU = {
    "name": "Axis Education Loan", "debt_type": "education_loan",
    "outstanding_amount": 310_000, "interest_rate": 11.0, "minimum_due": 0,
    "emi": 6_500, "overdue_cycles": 5,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_effective_rate_prefers_actual_over_default():
    assert get_effective_rate({"debt_type": "credit_card", "interest_rate": 42.0}) == (42.0, "actual")


def test_effective_rate_falls_back_to_default_table():
    rate, source = get_effective_rate({"debt_type": "credit_card", "interest_rate": None})
    assert (rate, source) == (DEFAULT_RATES["credit_card"], "estimated")


def test_effective_rate_unknown_debt_type_uses_other():
    rate, source = get_effective_rate({"debt_type": "does_not_exist", "interest_rate": None})
    assert (rate, source) == (DEFAULT_RATES["other"], "estimated")


@pytest.mark.parametrize(
    "debt_type,expected",
    [("credit_card", "revolving"), ("bnpl", "revolving"),
     ("home_loan", "emi"), ("education_loan", "emi"), ("car_loan", "emi")],
)
def test_debt_bucketing(debt_type, expected):
    assert get_debt_bucket({"debt_type": debt_type}) == expected


def test_emi_arrears_compound_over_overdue_cycles():
    none_overdue = calculate_emi_arrears(emi=6_500, annual_rate=11.0, overdue_cycles=0)
    some_overdue = calculate_emi_arrears(emi=6_500, annual_rate=11.0, overdue_cycles=5)
    assert none_overdue == 0
    assert some_overdue > 6_500 * 5           # compounding, not simple multiplication
    assert some_overdue == pytest.approx(33_404.75, abs=0.01)


def test_revolving_loss_is_one_month_of_interest():
    # 140000 * 42/1200 == 4900
    assert calculate_loss_if_not_paid(CARD) == pytest.approx(4_900.0)


def test_mandatory_payments_uses_emi_for_loans_and_min_due_for_cards():
    assert calculate_mandatory_payments([CARD, EDU]) == 7_000 + 6_500


def test_emi_debts_set_covers_expected_loan_types():
    assert {"home_loan", "personal_loan", "education_loan", "car_loan"} <= EMI_DEBTS
    assert "credit_card" not in EMI_DEBTS


# --------------------------------------------------------------------------- #
# Scenario selection
# --------------------------------------------------------------------------- #

def test_scenario_income_sufficient():
    out = rank_debts(debts=[CARD, EDU], monthly_income=100_000,
                     essential_expenses=40_000, emergency_fund=50_000)
    assert out["scenario"] == "income_sufficient"


def test_scenario_falls_back_to_emergency_fund():
    # 60k - 48k = 12k available, mandatory is 13.5k -> 1.5k must come from the fund
    out = rank_debts(debts=[CARD, EDU], monthly_income=60_000,
                     essential_expenses=48_000, emergency_fund=5_000)
    assert out["scenario"] == "use_emergency_fund"


def test_scenario_cannot_cover_mandatory():
    out = rank_debts(debts=[CARD, EDU], monthly_income=30_000,
                     essential_expenses=29_000, emergency_fund=0)
    assert out["scenario"] == "cannot_cover_mandatory"


def test_ranking_annotates_every_debt_with_loss():
    out = rank_debts(debts=[CARD, EDU], monthly_income=100_000,
                     essential_expenses=40_000, emergency_fund=50_000)
    assert len(out["ranked_debts"]) == 2
    assert all("loss_if_not_paid" in d for d in out["ranked_debts"])


# --------------------------------------------------------------------------- #
# Allocation — golden plan
# --------------------------------------------------------------------------- #

def test_allocation_golden_plan_draws_exactly_the_shortfall_from_the_fund():
    ranked = rank_debts(debts=[CARD, EDU], monthly_income=60_000,
                        essential_expenses=48_000, emergency_fund=5_000)
    plan = allocate_payments(
        monthly_income=60_000, essential_expenses=48_000, emergency_fund=5_000,
        scenario=ranked["scenario"], debts=ranked["ranked_debts"],
    )
    assert plan["emergency_fund_used"] == pytest.approx(1_500.0)
    assert plan["remaining_emergency_fund"] == pytest.approx(3_500.0)
    assert plan["remaining_income"] == pytest.approx(0.0)

    by_name = {d["name"]: d for d in plan["debts"]}
    assert by_name["HDFC Card"]["total_paid"] == pytest.approx(7_000.0)
    assert by_name["HDFC Card"]["outstanding_amount"] == pytest.approx(133_000.0)
    assert by_name["Axis Education Loan"]["total_paid"] == pytest.approx(6_500.0)
    assert by_name["Axis Education Loan"]["outstanding_amount"] == pytest.approx(303_500.0)


def test_allocation_never_pays_more_than_the_debt_or_leaves_negatives():
    ranked = rank_debts(debts=[CARD, EDU], monthly_income=200_000,
                        essential_expenses=40_000, emergency_fund=100_000)
    plan = allocate_payments(
        monthly_income=200_000, essential_expenses=40_000, emergency_fund=100_000,
        scenario=ranked["scenario"], debts=ranked["ranked_debts"],
    )
    assert plan["remaining_income"] >= 0
    assert plan["remaining_emergency_fund"] >= 0
    for d in plan["debts"]:
        assert d["outstanding_amount"] >= 0
        assert d["total_paid"] >= 0


def test_allocation_does_not_mutate_the_caller_s_debts():
    """allocate_payments deep-copies; the inputs must survive unchanged."""
    debts = [dict(CARD), dict(EDU)]
    before = [dict(d) for d in debts]
    ranked = rank_debts(debts=debts, monthly_income=100_000,
                        essential_expenses=40_000, emergency_fund=50_000)
    allocate_payments(monthly_income=100_000, essential_expenses=40_000,
                      emergency_fund=50_000, scenario=ranked["scenario"],
                      debts=ranked["ranked_debts"])
    assert [{k: d[k] for k in before[0]} for d in debts] == before


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key_and_full_contract(indebted_profile):
    patch = debt_trap_node(new_state(indebted_profile))
    assert set(patch) == {"debt_trap_result"}

    result = patch["debt_trap_result"]
    assert set(result) == {
        "scenario", "ranked_debts", "allocation", "total_debt", "debt_to_income",
    }
    assert result["total_debt"] == pytest.approx(450_000.0)
    assert result["scenario"] in {
        "income_sufficient", "use_emergency_fund", "cannot_cover_mandatory",
    }


def test_node_handles_a_user_with_no_debt(salaried_profile):
    result = debt_trap_node(new_state(salaried_profile))["debt_trap_result"]
    assert result["scenario"] == "no_debt"
    assert result["ranked_debts"] == []
    assert result["allocation"] is None
    assert result["total_debt"] == 0


def test_node_accepts_debt_with_no_interest_rate():
    """A Debt built without a rate must flow through the DEFAULT_RATES fallback."""
    profile = UserProfile(
        user_id="t", monthly_income=80_000, essential_expenses=30_000,
        existing_emergency_fund=20_000,
        debts=[Debt(name="Unknown Card", debt_type="credit_card",
                    outstanding_amount=50_000, minimum_due=2_500)],
    )
    result = debt_trap_node(new_state(profile))["debt_trap_result"]
    assert result["scenario"] == "income_sufficient"
    assert result["ranked_debts"][0]["loss_if_not_paid"] > 0
