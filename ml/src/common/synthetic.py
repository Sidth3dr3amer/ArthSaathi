"""
Synthetic transaction generator for the Behavioral Council.

The repo has no transaction history, and the behavioural agents are meaningless
without one -- you cannot detect a month-end spending spike in a dict of monthly
totals. This generates 24 months of plausible daily transactions with specific
behavioural signals *deliberately planted*, so the agents can be tested against
a known ground truth rather than against whatever a random walk produced.

Planted signals (each has a matching detector in the Behavioral Council):

  month_end_splurge   dining/entertainment spikes in the last 5 days of a month
  salary_day_splurge  discretionary spike within 2 days of salary credit
  subscription_creep  new recurring subscriptions accumulate and are never cancelled
  lifestyle_inflation discretionary spend rises with income rather than saving
  impulse_clusters    bursts of online shopping on consecutive days
  no_emergency_saving little or nothing routed to savings in surplus months

Seeded, so the dataset is identical on every machine and in CI.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

from . import config

SALARY_DAY = 1

#: category -> (base monthly amount as a share of income, is_discretionary)
CATEGORY_MIX: dict[str, tuple[float, bool]] = {
    "rent": (0.28, False),
    "groceries": (0.10, False),
    "utility_bills": (0.035, False),
    "fuel": (0.04, False),
    "healthcare": (0.02, False),
    "dining": (0.07, True),
    "entertainment": (0.035, True),
    "online_shopping": (0.06, True),
    "offline_retail": (0.04, True),
    "travel": (0.03, True),
    "subscriptions": (0.015, True),
}

MERCHANTS = {
    "rent": ["Landlord Transfer"],
    "groceries": ["BigBasket", "DMart", "Local Kirana", "Blinkit"],
    "utility_bills": ["MSEB", "Airtel Postpaid", "Mahanagar Gas"],
    "fuel": ["HP Petrol Pump", "Indian Oil"],
    "healthcare": ["Apollo Pharmacy", "Practo"],
    "dining": ["Swiggy", "Zomato", "Barbeque Nation", "Third Wave Coffee"],
    "entertainment": ["PVR Cinemas", "BookMyShow", "Spotify"],
    "online_shopping": ["Amazon", "Flipkart", "Myntra", "Ajio"],
    "offline_retail": ["Reliance Trends", "Shoppers Stop"],
    "travel": ["IRCTC", "MakeMyTrip", "Uber", "IndiGo"],
    "subscriptions": ["Netflix", "Prime Video", "Hotstar", "Gym Membership", "iCloud"],
}

SUBSCRIPTION_LADDER = [
    ("Netflix", 649), ("Prime Video", 179), ("Spotify", 119),
    ("Hotstar", 299), ("Gym Membership", 1_500), ("iCloud", 219),
]


def generate_transactions(
    months: int = 24,
    monthly_income: float = 90_000,
    income_growth_annual: float = 0.10,
    seed: int = 42,
    start: dt.date | None = None,
) -> dict[str, Any]:
    """
    Build a transaction history with known behavioural signals.

    Returns `{"meta": {...}, "transactions": [...]}` where meta records exactly
    which signals were planted, so tests assert against ground truth.
    """
    rng = random.Random(seed)
    start = start or (dt.date.today().replace(day=1) - dt.timedelta(days=30 * months))
    start = start.replace(day=1)

    transactions: list[dict[str, Any]] = []
    active_subscriptions: list[tuple[str, int]] = []

    for month_index in range(months):
        year = start.year + (start.month - 1 + month_index) // 12
        month = (start.month - 1 + month_index) % 12 + 1
        month_start = dt.date(year, month, 1)
        next_month = dt.date(year + (month == 12), month % 12 + 1, 1)
        days_in_month = (next_month - month_start).days

        # Income grows over time; the planted lifestyle-inflation signal is that
        # discretionary spend tracks it rather than the surplus being saved.
        income = monthly_income * (1 + income_growth_annual) ** (month_index / 12)
        inflation_factor = 1 + 0.45 * (income / monthly_income - 1)

        transactions.append({
            "date": month_start.isoformat(),
            "amount": round(income, 2),
            "category": "income",
            "merchant": "Salary Credit",
            "direction": "credit",
            "is_recurring": True,
        })

        # subscription_creep: a new subscription roughly every 4 months, never cancelled
        if month_index % 4 == 0 and len(active_subscriptions) < len(SUBSCRIPTION_LADDER):
            active_subscriptions.append(SUBSCRIPTION_LADDER[len(active_subscriptions)])

        for name, amount in active_subscriptions:
            transactions.append({
                "date": dt.date(year, month, min(5, days_in_month)).isoformat(),
                "amount": float(amount),
                "category": "subscriptions",
                "merchant": name,
                "direction": "debit",
                "is_recurring": True,
            })

        for category, (share, discretionary) in CATEGORY_MIX.items():
            if category == "subscriptions":
                continue

            budget = income * share
            if discretionary:
                budget *= inflation_factor

            if category == "rent":
                transactions.append({
                    "date": dt.date(year, month, 2).isoformat(),
                    "amount": round(budget, 2),
                    "category": "rent",
                    "merchant": "Landlord Transfer",
                    "direction": "debit",
                    "is_recurring": True,
                })
                continue

            n = rng.randint(3, 9) if discretionary else rng.randint(2, 5)
            for _ in range(n):
                day = rng.randint(1, days_in_month)
                weight = 1.0

                if discretionary and category in ("dining", "entertainment"):
                    # month_end_splurge
                    if day > days_in_month - 5:
                        weight *= rng.uniform(1.8, 2.6)
                    # salary_day_splurge
                    if abs(day - SALARY_DAY) <= 2:
                        weight *= rng.uniform(1.5, 2.2)

                amount = budget / n * weight * rng.uniform(0.7, 1.3)
                transactions.append({
                    "date": dt.date(year, month, day).isoformat(),
                    "amount": round(amount, 2),
                    "category": category,
                    "merchant": rng.choice(MERCHANTS[category]),
                    "direction": "debit",
                    "is_recurring": False,
                })

            # impulse_clusters: consecutive-day online shopping bursts
            if category == "online_shopping" and rng.random() < 0.45:
                burst_start = rng.randint(1, max(1, days_in_month - 3))
                for offset in range(rng.randint(2, 3)):
                    day = min(burst_start + offset, days_in_month)
                    transactions.append({
                        "date": dt.date(year, month, day).isoformat(),
                        "amount": round(budget * rng.uniform(0.25, 0.5), 2),
                        "category": "online_shopping",
                        "merchant": rng.choice(MERCHANTS["online_shopping"]),
                        "direction": "debit",
                        "is_recurring": False,
                    })

        # no_emergency_saving: a token transfer, far below the surplus
        if rng.random() < 0.35:
            transactions.append({
                "date": dt.date(year, month, min(28, days_in_month)).isoformat(),
                "amount": round(income * rng.uniform(0.01, 0.03), 2),
                "category": "savings",
                "merchant": "Recurring Deposit",
                "direction": "debit",
                "is_recurring": False,
            })

    transactions.sort(key=lambda t: t["date"])

    return {
        "meta": {
            "generated_by": "ml.src.common.synthetic.generate_transactions",
            "seed": seed,
            "months": months,
            "starting_monthly_income": monthly_income,
            "income_growth_annual": income_growth_annual,
            "synthetic": True,
            "planted_signals": [
                "month_end_splurge",
                "salary_day_splurge",
                "subscription_creep",
                "lifestyle_inflation",
                "impulse_clusters",
                "no_emergency_saving",
            ],
            "warning": "Synthetic data for development and demo. Not real user activity.",
        },
        "transactions": transactions,
    }


def load_transactions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the seeded dataset, returning [] when it is absent."""
    path = Path(path or config.TRANSACTIONS_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("transactions", [])
    except json.JSONDecodeError:
        return []


def write_dataset(path: Path | None = None, **kwargs: Any) -> Path:
    """Generate and persist the dataset."""
    path = Path(path or config.TRANSACTIONS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(generate_transactions(**kwargs), indent=1), encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    written = write_dataset()
    print(f"wrote {written}")
