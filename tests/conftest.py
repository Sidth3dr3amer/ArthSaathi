"""
Shared pytest fixtures.

Design rule: no test in this suite may require a network call or a live database
unless it is marked `@pytest.mark.live`. Agent cores are pure, so the default
suite runs fully offline and fast.
"""

from __future__ import annotations

import pytest

from ml.src.schemas.profile import Debt, Goal, UserProfile


@pytest.fixture
def salaried_profile() -> UserProfile:
    """A mid-income salaried user with one dependant. The default happy path."""
    return UserProfile(
        user_id="test-salaried",
        name="Test Salaried",
        age=28,
        job_type="salaried",
        dependents=1,
        monthly_income=90_000,
        essential_expenses=45_000,
        current_balance=120_000,
        existing_emergency_fund=50_000,
        has_health_insurance=True,
        monthly_spend={
            "online_shopping": 8_000,
            "groceries": 5_000,
            "dining": 4_000,
            "fuel": 3_000,
            "utility_bills": 3_000,
            "travel": 5_000,
        },
        max_annual_fee=2_000,
        prefer_cashback=True,
        travel_frequency="occasional",
    )


@pytest.fixture
def indebted_profile() -> UserProfile:
    """A user in deficit, carrying revolving and EMI debt. Exercises stress paths."""
    return UserProfile(
        user_id="test-indebted",
        name="Test Indebted",
        age=35,
        job_type="freelancer",
        dependents=3,
        monthly_income=60_000,
        essential_expenses=48_000,
        current_balance=8_000,
        existing_emergency_fund=5_000,
        has_health_insurance=False,
        debts=[
            Debt(
                name="HDFC Card",
                debt_type="credit_card",
                outstanding_amount=140_000,
                interest_rate=42.0,
                minimum_due=7_000,
                overdue_cycles=0,
            ),
            Debt(
                name="Axis Education Loan",
                debt_type="education_loan",
                outstanding_amount=310_000,
                interest_rate=11.0,
                emi=6_500,
                overdue_cycles=5,
            ),
        ],
        goals=[
            Goal(name="Home down payment", target_amount=1_500_000,
                 current_amount=200_000, target_months=48, priority="high"),
        ],
    )


#: 18 months of gently trending income with two expense spikes (months 5 and 11).
#: Long enough for Holt-Winters and SARIMAX to fit.
INCOME_HISTORY = [
    58_000, 60_000, 59_000, 61_000, 63_000, 62_000, 64_000, 66_000, 65_000,
    67_000, 69_000, 68_000, 70_000, 72_000, 71_000, 73_000, 75_000, 74_000,
]
EXPENSE_HISTORY = [
    44_000, 46_000, 45_000, 47_000, 52_000, 44_000, 45_000, 48_000, 46_000,
    45_000, 58_000, 47_000, 46_000, 49_000, 47_000, 48_000, 60_000, 49_000,
]


@pytest.fixture
def history_profile() -> UserProfile:
    """A user with enough history for the forecasting models to actually fit."""
    return UserProfile(
        user_id="test-history",
        name="Test History",
        monthly_income=74_000,
        essential_expenses=48_000,
        current_balance=120_000,
        dependents=1,
        income_history=list(INCOME_HISTORY),
        expense_history=list(EXPENSE_HISTORY),
        goals=[Goal(name="Emergency top-up", target_amount=300_000,
                    current_amount=120_000, target_months=24)],
    )


@pytest.fixture
def zero_profile() -> UserProfile:
    """All-zero boundary case. Nothing may divide by zero or return NaN."""
    return UserProfile(
        user_id="test-zero",
        name="Test Zero",
        monthly_income=0,
        essential_expenses=0,
        current_balance=0,
        existing_emergency_fund=0,
    )


@pytest.fixture
def stub_llm(monkeypatch):
    """
    Replace `ml.src.common.llm.chat` with a deterministic stub.

    Returns the list of recorded calls so a test can assert what the agent asked
    the model, without ever hitting a provider.
    """
    calls: list[dict] = []

    def _fake_chat(prompt: str, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return "STUBBED_LLM_RESPONSE"

    monkeypatch.setattr("ml.src.common.llm.chat", _fake_chat)
    return calls
