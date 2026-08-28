"""
Credit Card Intelligence -> Tier 1: User Understanding.

    User Profiler Agent -> Spending Analyzer Agent -> Financial Twin Agent

Tier 1 answers "who is this person, as a cardholder?" before any card is
scored. The three agents are deliberately separate because they answer
different questions from different inputs:

  User Profiler     demographics and stated preferences -> segment + travel profile
  Spending Analyzer observed spend -> annualised, card-relevant categories
  Financial Twin    both of the above -> the behavioural facts that decide whether
                    a card's headline benefits will actually be realised

The Financial Twin is the one that matters. A card offering 12 lounge visits is
worth nothing to someone who flies twice a year, and a 5% cashback card is worth
nothing to someone who revolves a balance at 42%. Tier 2 values benefits
*against this twin*, not against the card's brochure.

All three are pure functions. No I/O, no LLM.
"""

from __future__ import annotations

from typing import Any

from ..schemas.profile import UserProfile
from ..schemas.state import FinancialState

# --------------------------------------------------------------------------- #
# User Profiler
# --------------------------------------------------------------------------- #

#: Estimated annual flight segments implied by a stated travel frequency.
TRAVEL_SEGMENTS = {"none": 0, "occasional": 4, "frequent": 18}

#: Annual income bands (INR) -> the card tier a bank will realistically offer.
INCOME_BANDS: tuple[tuple[float, str], ...] = (
    (300_000, "Entry"),
    (600_000, "Beginner"),
    (1_200_000, "Mid-range"),
    (2_400_000, "Premium"),
    (float("inf"), "Super-premium"),
)


def _income_band(annual_income: float) -> str:
    for ceiling, label in INCOME_BANDS:
        if annual_income < ceiling:
            return label
    return "Super-premium"


def user_profiler_advisor(
    age: int,
    monthly_income: float,
    occupation: str | None = None,
    job_type: str = "salaried",
    city: str | None = None,
    travel_frequency: str = "none",
    international_spend_monthly: float = 0.0,
    dependents: int = 0,
) -> dict[str, Any]:
    """
    Build the cardholder profile. Pure and deterministic.

    Mirrors the doc's `{"income": ..., "travel_profile": ...}` output, expanded
    with the segment and life-stage signals Tier 2 needs.
    """
    annual_income = float(monthly_income) * 12

    if travel_frequency == "frequent" or international_spend_monthly > 15_000:
        travel_profile = "high"
    elif travel_frequency == "occasional" or international_spend_monthly > 3_000:
        travel_profile = "medium"
    else:
        travel_profile = "low"

    if age < 25:
        life_stage = "early_career"
    elif age < 35:
        life_stage = "establishing"
    elif age < 50:
        life_stage = "peak_earning"
    else:
        life_stage = "pre_retirement"

    return {
        "income": round(annual_income, 2),
        "monthly_income": round(float(monthly_income), 2),
        "income_band": _income_band(annual_income),
        "travel_profile": travel_profile,
        "estimated_flight_segments": TRAVEL_SEGMENTS.get(travel_frequency, 0),
        "has_forex_exposure": bool(international_spend_monthly > 0),
        "annual_international_spend": round(float(international_spend_monthly) * 12, 2),
        "age": int(age),
        "life_stage": life_stage,
        "occupation": occupation,
        "job_type": job_type,
        "city": city,
        "dependents": int(dependents),
        "salaried": job_type in ("salaried", "govt"),
    }


# --------------------------------------------------------------------------- #
# Spending Analyzer
# --------------------------------------------------------------------------- #

#: Raw spend categories -> the buckets card reward tables are written against.
CATEGORY_MAP: dict[str, str] = {
    "dining": "dining",
    "entertainment": "dining",
    "fuel": "fuel",
    "transport": "fuel",
    "travel": "travel",
    "international": "international",
    "online_shopping": "online",
    "offline_retail": "offline",
    "groceries": "grocery",
    "utility_bills": "utility",
    "rent": "rent",
    "housing": "rent",
    "subscriptions": "online",
    "healthcare": "other",
    "education": "other",
    "insurance": "other",
    "others": "other",
}

#: Categories most cards explicitly exclude from reward earning.
TYPICALLY_EXCLUDED = {"rent", "utility"}


def spending_analyzer_advisor(
    monthly_spend: dict[str, float],
    transactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Map raw spend into card-reward buckets and annualise. Pure and deterministic.

    When a transaction history is supplied it takes precedence over the stated
    monthly figures, because observed spend beats remembered spend.
    """
    source = "stated"
    spend: dict[str, float] = {k: float(v) for k, v in (monthly_spend or {}).items()}

    if transactions:
        months: set[str] = set()
        observed: dict[str, float] = {}
        for txn in transactions:
            if txn.get("direction") != "debit":
                continue
            category = txn.get("category")
            if category in (None, "savings", "income"):
                continue
            months.add(str(txn.get("date", ""))[:7])
            observed[category] = observed.get(category, 0.0) + float(txn.get("amount", 0))
        if observed and months:
            spend = {k: v / len(months) for k, v in observed.items()}
            source = f"observed over {len(months)} months"

    buckets: dict[str, float] = {}
    unmapped: list[str] = []
    for category, amount in spend.items():
        bucket = CATEGORY_MAP.get(category)
        if bucket is None:
            unmapped.append(category)
            bucket = "other"
        buckets[bucket] = buckets.get(bucket, 0.0) + float(amount)

    total_monthly = sum(buckets.values())
    rewardable = sum(v for k, v in buckets.items() if k not in TYPICALLY_EXCLUDED)

    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "source": source,
        "buckets_monthly": {k: round(v, 2) for k, v in buckets.items()},
        "buckets_annual": {k: round(v * 12, 2) for k, v in buckets.items()},
        "total_monthly": round(total_monthly, 2),
        "total_annual": round(total_monthly * 12, 2),
        "rewardable_monthly": round(rewardable, 2),
        "rewardable_annual": round(rewardable * 12, 2),
        "excluded_share": round(
            1 - (rewardable / total_monthly), 4
        ) if total_monthly > 0 else 0.0,
        "dominant_category": ranked[0][0] if ranked else None,
        "ranked_categories": [
            {"category": k, "monthly": round(v, 2),
             "share": round(v / total_monthly, 4) if total_monthly else 0.0}
            for k, v in ranked
        ],
        "unmapped_categories": sorted(unmapped),
    }


# --------------------------------------------------------------------------- #
# Financial Twin
# --------------------------------------------------------------------------- #

#: Below this, a user cannot comfortably absorb a premium annual fee.
FEE_COMFORT_MULTIPLE = 0.02        # annual fee as a share of annual income

#: Rate above which carrying a balance destroys any reward earned.
REVOLVING_RATE_THRESHOLD = 24.0


def financial_twin_advisor(
    profiler: dict[str, Any],
    spending: dict[str, Any],
    monthly_surplus: float = 0.0,
    revolving_debt: float = 0.0,
    highest_debt_rate: float = 0.0,
    existing_cards: int = 0,
    credit_utilisation: float = 0.0,
) -> dict[str, Any]:
    """
    The behavioural model Tier 2 values benefits against. Pure and deterministic.

    The central judgement here is `rewards_are_real`: someone revolving a balance
    at 42% is losing far more to interest than any reward rate returns, so their
    card decision is about cost, not rewards. Marking that explicitly stops
    Tier 4's Cashback Expert from confidently recommending the wrong thing.
    """
    annual_income = float(profiler.get("income", 0) or 0)
    total_annual_spend = float(spending.get("total_annual", 0) or 0)

    revolves = bool(revolving_debt > 0 and highest_debt_rate >= REVOLVING_RATE_THRESHOLD)
    annual_interest_cost = revolving_debt * (highest_debt_rate / 100) if revolves else 0.0

    fee_comfort = annual_income * FEE_COMFORT_MULTIPLE
    if monthly_surplus <= 0:
        fee_tolerance = 0.0
    else:
        fee_tolerance = min(fee_comfort, monthly_surplus * 3)

    segments = int(profiler.get("estimated_flight_segments", 0) or 0)
    # A lounge visit needs a flight; realistically one visit per segment at most,
    # and travellers do not use every eligible visit.
    realistic_lounge_visits = int(round(segments * 0.75))

    return {
        "annual_income": round(annual_income, 2),
        "annual_spend": round(total_annual_spend, 2),
        "spend_to_income": round(
            total_annual_spend / annual_income, 4
        ) if annual_income > 0 else 0.0,
        "monthly_surplus": round(float(monthly_surplus), 2),
        "revolves_balance": revolves,
        "annual_interest_cost": round(annual_interest_cost, 2),
        "rewards_are_real": not revolves,
        "fee_tolerance": round(fee_tolerance, 2),
        "can_absorb_premium_fee": bool(fee_tolerance >= 5_000),
        "realistic_lounge_visits": realistic_lounge_visits,
        "travel_profile": profiler.get("travel_profile", "low"),
        "forex_exposure_annual": float(profiler.get("annual_international_spend", 0) or 0),
        "existing_cards": int(existing_cards),
        "credit_utilisation": round(float(credit_utilisation), 4),
        "approval_headroom": bool(credit_utilisation < 0.5 and existing_cards < 5),
        "warnings": [
            w for w in (
                (
                    f"Carrying Rs {revolving_debt:,.0f} at {highest_debt_rate:.0f}% costs "
                    f"about Rs {annual_interest_cost:,.0f} a year, which exceeds what "
                    "any reward rate returns. Card choice here is about cost, not rewards."
                ) if revolves else None,
                (
                    "No monthly surplus, so any annual fee is funded from savings."
                ) if monthly_surplus <= 0 else None,
                (
                    f"Credit utilisation at {credit_utilisation:.0%} may reduce approval "
                    "odds on premium cards."
                ) if credit_utilisation >= 0.5 else None,
            ) if w
        ],
    }


# --------------------------------------------------------------------------- #
# Tier entry point
# --------------------------------------------------------------------------- #

def run_tier1(
    profile: UserProfile,
    transactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the three Tier 1 agents in order."""
    rates = [d.interest_rate or 0 for d in profile.debts]
    revolving = sum(
        d.outstanding_amount for d in profile.debts if d.debt_type == "credit_card"
    )

    profiler = user_profiler_advisor(
        age=profile.age,
        monthly_income=profile.monthly_income,
        occupation=profile.occupation,
        job_type=profile.job_type,
        city=profile.state,
        travel_frequency=profile.travel_frequency,
        international_spend_monthly=float(profile.monthly_spend.get("international", 0)),
        dependents=profile.dependents,
    )
    spending = spending_analyzer_advisor(profile.monthly_spend, transactions)
    twin = financial_twin_advisor(
        profiler=profiler,
        spending=spending,
        monthly_surplus=profile.monthly_surplus,
        revolving_debt=revolving,
        highest_debt_rate=max(rates) if rates else 0.0,
        existing_cards=sum(1 for d in profile.debts if d.debt_type == "credit_card"),
        credit_utilisation=min(profile.debt_to_income * 2, 1.0),
    )

    return {"profiler": profiler, "spending": spending, "twin": twin}


def tier1_node(
    state: FinancialState, transactions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Adapter. Writes one result key."""
    if transactions is None:
        transactions = state.get("transactions")
    return {"card_tier1_result": run_tier1(state["profile"], transactions)}
