"""Shared fixtures for the credit-card Tier 1-6 tests."""

from __future__ import annotations

import pytest

from ml.src.cards.tier1_profiler import run_tier1
from ml.src.cards.tier2_evaluation import evaluate_all
from ml.src.cards.tier3_twin import simulate_all
from ml.src.cards.tier4_experts import deliberate
from ml.src.councils.growth.credit_card import (
    analyze_spend_profile,
    filter_eligible_cards,
    load_card_database,
    profile_to_engine_dict,
)
from ml.src.schemas.profile import Debt, UserProfile


@pytest.fixture(scope="session")
def card_db():
    return load_card_database()


@pytest.fixture
def card_user() -> UserProfile:
    """A salaried mid-income user who travels occasionally and pays in full."""
    return UserProfile(
        user_id="cards", name="Rahul", age=32, job_type="salaried",
        monthly_income=100_000, essential_expenses=45_000, max_annual_fee=5_000,
        travel_frequency="occasional", prefer_cashback=True,
        monthly_spend={
            "online_shopping": 9_000, "groceries": 6_000, "dining": 5_000,
            "fuel": 3_000, "utility_bills": 3_000, "travel": 6_000,
            "international": 2_500, "offline_retail": 4_000,
        },
    )


@pytest.fixture
def revolver() -> UserProfile:
    """A user carrying a card balance at 42% -- rewards are not real for them."""
    return UserProfile(
        user_id="revolver", age=30, job_type="salaried",
        monthly_income=60_000, essential_expenses=55_000, max_annual_fee=5_000,
        monthly_spend={"online_shopping": 8_000, "groceries": 6_000},
        debts=[Debt(name="Card", debt_type="credit_card",
                    outstanding_amount=180_000, interest_rate=42.0, minimum_due=9_000)],
    )


@pytest.fixture
def frequent_flyer() -> UserProfile:
    return UserProfile(
        user_id="flyer", age=38, job_type="salaried",
        monthly_income=300_000, essential_expenses=90_000, max_annual_fee=20_000,
        travel_frequency="frequent",
        monthly_spend={"travel": 40_000, "international": 25_000, "dining": 15_000},
    )


@pytest.fixture
def tier1(card_user):
    return run_tier1(card_user)


@pytest.fixture
def engine_bits(card_user, card_db):
    engine_profile = profile_to_engine_dict(card_user)
    spend_analysis = analyze_spend_profile(engine_profile)
    eligible, rejected = filter_eligible_cards(engine_profile, card_db)
    return engine_profile, spend_analysis, eligible, rejected


@pytest.fixture
def evaluations(engine_bits, tier1):
    engine_profile, spend_analysis, eligible, _ = engine_bits
    return evaluate_all(eligible, engine_profile, spend_analysis, tier1["twin"])


@pytest.fixture
def simulations(evaluations, tier1):
    return simulate_all(evaluations, tier1["twin"])


@pytest.fixture
def deliberation(evaluations, tier1):
    return deliberate(evaluations, tier1["twin"])
