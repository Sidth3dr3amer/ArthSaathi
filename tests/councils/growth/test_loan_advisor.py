"""Growth Council -> Loan Advisor Agent."""

from __future__ import annotations

import json

import pytest

from ml.src.councils.growth.loan_advisor import (
    ASSUMED_INVESTMENT_RETURN,
    LENDER_FOIR,
    PRUDENT_FOIR,
    emi,
    loan_advisor_advisor,
    loan_advisor_node,
    max_principal,
)
from ml.src.schemas.profile import Debt, UserProfile
from ml.src.schemas.state import new_state


def advise(**overrides):
    base = dict(monthly_income=120_000, existing_obligations=0.0,
                loan_type="personal_loan")
    base.update(overrides)
    return loan_advisor_advisor(**base)


# --------------------------------------------------------------------------- #
# EMI maths
# --------------------------------------------------------------------------- #

def test_emi_matches_the_standard_formula():
    # 10 lakh, 9%, 20 years -> about Rs 8,997
    assert emi(1_000_000, 9.0, 240) == pytest.approx(8_997, abs=5)


def test_zero_rate_loan_is_straight_division():
    assert emi(120_000, 0.0, 12) == pytest.approx(10_000)


def test_emi_of_zero_tenure_is_zero():
    assert emi(100_000, 10.0, 0) == 0.0


def test_max_principal_inverts_emi():
    principal = max_principal(emi(500_000, 12.0, 60), 12.0, 60)
    assert principal == pytest.approx(500_000, rel=1e-6)


def test_max_principal_of_no_capacity_is_zero():
    assert max_principal(0, 12.0, 60) == 0.0


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #

def test_lender_ceiling_exceeds_the_prudent_figure():
    out = advise()
    assert out["lender_max_principal"] > out["prudent_max_principal"]
    assert out["lender_max_emi"] == pytest.approx(120_000 * LENDER_FOIR)
    assert out["prudent_max_emi"] == pytest.approx(120_000 * PRUDENT_FOIR)


def test_existing_obligations_reduce_headroom():
    clean = advise(existing_obligations=0)["lender_max_principal"]
    burdened = advise(existing_obligations=30_000)["lender_max_principal"]
    assert burdened < clean


def test_fully_committed_income_leaves_no_capacity():
    out = advise(existing_obligations=100_000)
    assert out["lender_max_principal"] == 0
    assert out["status"] == "Over-leveraged"


@pytest.mark.parametrize(
    "obligations,expected",
    [(0, "Ample headroom"), (30_000, "Comfortable"),
     (54_000, "Stretched"), (70_000, "Over-leveraged")],
)
def test_status_bands(obligations, expected):
    assert advise(existing_obligations=obligations)["status"] == expected


def test_zero_income_does_not_divide_by_zero():
    out = advise(monthly_income=0)
    assert out["current_foir"] == 0.0
    assert out["lender_max_principal"] == 0


# --------------------------------------------------------------------------- #
# Request assessment
# --------------------------------------------------------------------------- #

def test_affordable_request_is_approved():
    out = advise(requested_amount=300_000)
    assert out["assessment"]["verdict"] == "affordable"
    assert out["assessment"]["within_prudent_limit"] is True


def test_request_between_prudent_and_lender_limits_is_flagged_as_stretched():
    out = advise()
    between = (out["prudent_max_principal"] + out["lender_max_principal"]) / 2
    assessed = advise(requested_amount=between)["assessment"]
    assert assessed["verdict"] == "approvable_but_stretched"
    assert assessed["within_lender_limit"] is True
    assert assessed["within_prudent_limit"] is False


def test_request_beyond_the_lender_ceiling_is_rejected():
    out = advise(requested_amount=50_000_000)
    assert out["assessment"]["verdict"] == "likely_rejected"


def test_assessment_reports_total_interest():
    out = advise(requested_amount=500_000)["assessment"]
    assert out["total_interest"] == pytest.approx(
        out["total_repayment"] - out["requested_amount"], abs=1
    )
    assert out["total_interest"] > 0


def test_no_request_means_no_assessment():
    assert advise()["assessment"] == {}


# --------------------------------------------------------------------------- #
# Prepay vs invest
# --------------------------------------------------------------------------- #

def test_expensive_debt_should_be_prepaid():
    out = advise(highest_debt_rate=42.0, investable_surplus=20_000)
    assert out["prepay_vs_invest"]["recommendation"] == "prepay_debt"
    assert out["prepay_vs_invest"]["spread"] == pytest.approx(42.0 - ASSUMED_INVESTMENT_RETURN)


def test_cheap_debt_should_not_be_prepaid():
    out = advise(highest_debt_rate=8.5, investable_surplus=20_000)
    assert out["prepay_vs_invest"]["recommendation"] == "invest_surplus"


def test_the_invest_case_acknowledges_that_the_return_is_not_guaranteed():
    """A saving is certain; a market return is not. The rationale must say so."""
    rationale = advise(highest_debt_rate=8.5, investable_surplus=20_000)[
        "prepay_vs_invest"]["rationale"]
    assert "not guaranteed" in rationale


def test_no_surplus_means_no_prepay_advice():
    assert advise(highest_debt_rate=42.0, investable_surplus=0)["prepay_vs_invest"] == {}


def test_no_debt_means_no_prepay_advice():
    assert advise(investable_surplus=20_000)["prepay_vs_invest"] == {}


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_writes_one_key(indebted_profile):
    patch = loan_advisor_node(new_state(indebted_profile))
    assert set(patch) == {"loan_advisor_result"}
    json.dumps(patch)


def test_node_derives_obligations_and_worst_rate_from_the_profile():
    profile = UserProfile(
        user_id="n", monthly_income=100_000,
        debts=[
            Debt(name="Card", debt_type="credit_card",
                 outstanding_amount=100_000, interest_rate=42.0, minimum_due=5_000),
            Debt(name="Home", debt_type="home_loan",
                 outstanding_amount=3_000_000, interest_rate=8.5, emi=25_000),
        ],
    )
    result = loan_advisor_node(new_state(profile))["loan_advisor_result"]
    assert result["current_foir"] == pytest.approx(0.30)
    assert result["prepay_vs_invest"]["highest_debt_rate"] == 42.0


def test_node_survives_the_zero_profile(zero_profile):
    result = loan_advisor_node(new_state(zero_profile))["loan_advisor_result"]
    assert result["lender_max_principal"] == 0
