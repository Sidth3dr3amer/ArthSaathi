"""
Transaction feature extraction shared by the Behavioral Council.

All four behavioural agents reason over the same derived features, so they are
computed once here rather than four slightly-different times. Everything is a
pure function over a list of transaction dicts.

Transaction shape (see `ml.src.common.synthetic`):
    {"date": "YYYY-MM-DD", "amount": float, "category": str,
     "merchant": str, "direction": "debit"|"credit", "is_recurring": bool}
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

DISCRETIONARY = {
    "dining", "entertainment", "online_shopping", "offline_retail",
    "travel", "subscriptions",
}
ESSENTIAL = {"rent", "housing", "groceries", "utility_bills", "fuel", "healthcare", "education"}

#: Days from month end that count as "month end", and days from the 1st that
#: count as "just after salary".
MONTH_END_WINDOW = 5
SALARY_WINDOW = 2


def _day(txn: dict[str, Any]) -> int:
    return int(txn["date"][8:10])


def _month(txn: dict[str, Any]) -> str:
    return txn["date"][:7]


def _days_in_month(month_key: str) -> int:
    year, month = int(month_key[:4]), int(month_key[5:7])
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return (nxt - dt.date(year, month, 1)).days


def debits(transactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in transactions if t.get("direction") == "debit"]


def credits(transactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in transactions if t.get("direction") == "credit"]


def monthly_totals(
    transactions: Iterable[dict[str, Any]], categories: set[str] | None = None
) -> dict[str, float]:
    """Total debit per month, optionally restricted to categories."""
    totals: dict[str, float] = defaultdict(float)
    for txn in debits(transactions):
        if categories and txn["category"] not in categories:
            continue
        totals[_month(txn)] += txn["amount"]
    return dict(sorted(totals.items()))


def monthly_income(transactions: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for txn in credits(transactions):
        totals[_month(txn)] += txn["amount"]
    return dict(sorted(totals.items()))


def category_totals(transactions: Iterable[dict[str, Any]]) -> dict[str, float]:
    totals: Counter = Counter()
    for txn in debits(transactions):
        totals[txn["category"]] += txn["amount"]
    return dict(totals.most_common())


def _avg(values: list[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _timing_buckets(
    transactions: Iterable[dict[str, Any]], categories: set[str]
) -> dict[str, Any]:
    """Bucket transaction sizes by position in the month, for given categories."""
    early, middle, late, post_salary = [], [], [], []
    for txn in debits(transactions):
        if txn["category"] not in categories:
            continue
        day = _day(txn)
        limit = _days_in_month(_month(txn))
        if day <= SALARY_WINDOW + 1:
            post_salary.append(txn["amount"])
        if day <= 10:
            early.append(txn["amount"])
        elif day <= limit - MONTH_END_WINDOW:
            middle.append(txn["amount"])
        else:
            late.append(txn["amount"])

    mid_avg, late_avg, salary_avg = _avg(middle), _avg(late), _avg(post_salary)
    return {
        "early_month_avg": _avg(early),
        "mid_month_avg": mid_avg,
        "month_end_avg": late_avg,
        "post_salary_avg": salary_avg,
        "month_end_ratio": round(late_avg / mid_avg, 3) if mid_avg else 0.0,
        "salary_day_ratio": round(salary_avg / mid_avg, 3) if mid_avg else 0.0,
        "sample_size": len(early) + len(middle) + len(late),
        "counts": {
            "early": len(early), "middle": len(middle),
            "late": len(late), "post_salary": len(post_salary),
        },
    }


def timing_profile(transactions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """
    Transaction size by position in the month, blended and per category.

    The ratio of month-end to mid-month spend is the signature of present bias:
    money behaves differently depending on which mental "pot" it sits in.

    Reported per category as well as blended, because blending hides real
    effects -- a strong dining spike averages away against steady grocery
    spend, and "you overspend on dining at month end" is far more actionable
    than a diluted aggregate.
    """
    txns = list(transactions)
    blended = _timing_buckets(txns, DISCRETIONARY)
    present = {t["category"] for t in debits(txns)} & DISCRETIONARY
    blended["by_category"] = {
        category: _timing_buckets(txns, {category}) for category in sorted(present)
    }
    return blended


def recurring_merchants(transactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merchants charging in at least three distinct months."""
    seen: dict[str, set[str]] = defaultdict(set)
    amounts: dict[str, list[float]] = defaultdict(list)
    category: dict[str, str] = {}
    for txn in debits(transactions):
        key = txn.get("merchant", "unknown")
        seen[key].add(_month(txn))
        amounts[key].append(txn["amount"])
        category[key] = txn["category"]

    out = []
    for merchant, months in seen.items():
        if len(months) < 3:
            continue
        out.append({
            "merchant": merchant,
            "category": category[merchant],
            "months_active": len(months),
            "first_month": min(months),
            "last_month": max(months),
            "monthly_amount": round(statistics.median(amounts[merchant]), 2),
            "annual_cost": round(statistics.median(amounts[merchant]) * 12, 2),
        })
    return sorted(out, key=lambda m: m["annual_cost"], reverse=True)


def subscription_growth(transactions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Distinct subscription merchants per month, first to last."""
    per_month: dict[str, set[str]] = defaultdict(set)
    for txn in debits(transactions):
        if txn["category"] == "subscriptions":
            per_month[_month(txn)].add(txn.get("merchant", "unknown"))

    months = sorted(per_month)
    if not months:
        return {"first": 0, "last": 0, "added": 0, "months": 0, "series": {}}
    return {
        "first": len(per_month[months[0]]),
        "last": len(per_month[months[-1]]),
        "added": len(per_month[months[-1]]) - len(per_month[months[0]]),
        "months": len(months),
        "series": {m: len(v) for m, v in sorted(per_month.items())},
    }


def impulse_clusters(
    transactions: Iterable[dict[str, Any]], category: str = "online_shopping",
    min_run: int = 2,
) -> list[dict[str, Any]]:
    """Runs of purchases in one category on consecutive days."""
    days = sorted({
        dt.date.fromisoformat(t["date"]) for t in debits(transactions)
        if t["category"] == category
    })
    clusters, run = [], []
    for day in days:
        if run and (day - run[-1]).days == 1:
            run.append(day)
            continue
        if len(run) >= min_run:
            clusters.append(run)
        run = [day]
    if len(run) >= min_run:
        clusters.append(run)

    amounts_by_day: dict[dt.date, float] = defaultdict(float)
    for txn in debits(transactions):
        if txn["category"] == category:
            amounts_by_day[dt.date.fromisoformat(txn["date"])] += txn["amount"]

    return [
        {
            "start": run[0].isoformat(),
            "end": run[-1].isoformat(),
            "days": len(run),
            "total": round(sum(amounts_by_day[d] for d in run), 2),
        }
        for run in clusters
    ]


def trend(series: dict[str, float]) -> float:
    """
    Fractional change from the first third of the series to the last third.

    Compares thirds rather than endpoints so a single unusual month does not
    masquerade as a trend.
    """
    values = list(series.values())
    if len(values) < 6:
        return 0.0
    third = max(1, len(values) // 3)
    head = statistics.mean(values[:third])
    tail = statistics.mean(values[-third:])
    return round((tail - head) / head, 4) if head else 0.0


def savings_rate(transactions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Share of income routed to savings, and how much surplus went unsaved."""
    income = monthly_income(transactions)
    saved = monthly_totals(transactions, {"savings"})
    spent = {
        m: v for m, v in monthly_totals(transactions).items()
    }

    per_month = {}
    for month, earned in income.items():
        outgoing = spent.get(month, 0.0)
        put_away = saved.get(month, 0.0)
        surplus = earned - (outgoing - put_away)
        per_month[month] = {
            "income": round(earned, 2),
            "spent": round(outgoing - put_away, 2),
            "saved": round(put_away, 2),
            "surplus": round(surplus, 2),
            "capture": round(put_away / surplus, 4) if surplus > 0 else 0.0,
        }

    captures = [m["capture"] for m in per_month.values() if m["surplus"] > 0]
    return {
        "per_month": per_month,
        "mean_capture": round(statistics.mean(captures), 4) if captures else 0.0,
        "months_with_surplus": len(captures),
        "months_saved_nothing": sum(
            1 for m in per_month.values() if m["surplus"] > 0 and m["saved"] == 0
        ),
    }


def extract_features(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything the Behavioral Council needs, computed once."""
    disc = monthly_totals(transactions, DISCRETIONARY)
    return {
        "transaction_count": len(transactions),
        "months": len(monthly_totals(transactions)),
        "monthly_income": monthly_income(transactions),
        "monthly_spend": monthly_totals(transactions),
        "discretionary_by_month": disc,
        "category_totals": category_totals(transactions),
        "timing": timing_profile(transactions),
        "recurring": recurring_merchants(transactions),
        "subscriptions": subscription_growth(transactions),
        "impulse_clusters": impulse_clusters(transactions),
        "income_trend": trend(monthly_income(transactions)),
        "discretionary_trend": trend(disc),
        "savings": savings_rate(transactions),
    }
