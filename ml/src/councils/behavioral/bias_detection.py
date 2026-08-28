"""
Behavioral Council -> Bias Detection Agent.

Detects behavioural patterns in transaction history and prices them.

Design stance: every finding must carry the evidence that produced it and an
annual rupee cost. "You have present bias" is an insult; "your dining spend runs
1.8x higher in the last five days of the month, which costs about Rs 14,000 a
year" is information someone can act on. The agent names the bias for the
record, but leads with the observation.

Detections are deterministic thresholds over `features.extract_features`, not an
LLM, so the same history always yields the same findings.
"""

from __future__ import annotations

from typing import Any

from ...schemas.state import FinancialState
from . import features as F

#: Minimum transactions in a bucket before a ratio means anything.
MIN_SAMPLE = 20

#: Ratio above which month-end / salary-day spending counts as a spike.
SPIKE_THRESHOLD = 1.35

#: Discretionary growth must outpace income growth by this much to count as
#: lifestyle inflation rather than ordinary price rises.
INFLATION_MARGIN = 1.5

#: Share of surplus that should be captured before saving looks adequate.
HEALTHY_CAPTURE = 0.20


def _finding(
    bias: str, label: str, detected: bool, strength: float,
    observation: str, evidence: dict[str, Any], annual_cost: float = 0.0,
) -> dict[str, Any]:
    return {
        "bias": bias,
        "label": label,
        "detected": detected,
        "strength": round(max(0.0, min(1.0, strength)), 3),
        "observation": observation,
        "evidence": evidence,
        "estimated_annual_cost": round(annual_cost, 2),
    }


def detect_present_bias(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Spending that spikes at month end or just after payday."""
    out = []
    timing = feats["timing"]
    months = max(feats["months"], 1)

    for category, stats in timing.get("by_category", {}).items():
        if stats["sample_size"] < MIN_SAMPLE:
            continue

        for kind, ratio_key, window_label in (
            ("month_end", "month_end_ratio", "the last 5 days of the month"),
            ("salary_day", "salary_day_ratio", "the 2 days after salary lands"),
        ):
            ratio = stats[ratio_key]
            if ratio < SPIKE_THRESHOLD:
                continue

            bucket = "late" if kind == "month_end" else "post_salary"
            excess_per_txn = stats[
                "month_end_avg" if kind == "month_end" else "post_salary_avg"
            ] - stats["mid_month_avg"]
            annual_cost = excess_per_txn * stats["counts"][bucket] / months * 12

            out.append(_finding(
                bias="present_bias" if kind == "salary_day" else "mental_accounting",
                label=f"{category.replace('_', ' ').title()} spikes in {window_label}",
                detected=True,
                strength=min((ratio - 1) / 1.5, 1.0),
                observation=(
                    f"{category.replace('_', ' ').title()} transactions average "
                    f"Rs {stats['month_end_avg' if kind == 'month_end' else 'post_salary_avg']:,.0f} "
                    f"during {window_label}, against Rs {stats['mid_month_avg']:,.0f} "
                    f"mid-month -- {ratio:.1f}x higher."
                ),
                evidence={
                    "category": category, "window": kind, "ratio": ratio,
                    "sample_size": stats["sample_size"],
                },
                annual_cost=max(annual_cost, 0.0),
            ))
    return out


def detect_lifestyle_inflation(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Discretionary spend rising faster than income."""
    income_trend = feats["income_trend"]
    disc_trend = feats["discretionary_trend"]

    if disc_trend <= 0 or income_trend <= 0:
        return []
    if disc_trend < income_trend * INFLATION_MARGIN:
        return []

    disc = feats["discretionary_by_month"]
    values = list(disc.values())
    third = max(1, len(values) // 3)
    baseline = sum(values[:third]) / third
    annual_cost = (disc_trend - income_trend) * baseline * 12

    return [_finding(
        bias="lifestyle_inflation",
        label="Spending is growing faster than income",
        detected=True,
        strength=min((disc_trend - income_trend) / 0.3, 1.0),
        observation=(
            f"Discretionary spending rose {disc_trend:.0%} over the period while "
            f"income rose {income_trend:.0%}. Raises are being absorbed by "
            f"lifestyle rather than savings."
        ),
        evidence={"income_trend": income_trend, "discretionary_trend": disc_trend},
        annual_cost=max(annual_cost, 0.0),
    )]


def detect_subscription_creep(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Recurring charges accumulating and never being cancelled."""
    subs = feats["subscriptions"]
    if subs["added"] < 2:
        return []

    sub_cost = sum(
        m["annual_cost"] for m in feats["recurring"] if m["category"] == "subscriptions"
    )
    return [_finding(
        bias="status_quo_bias",
        label="Subscriptions accumulate but are never cancelled",
        detected=True,
        strength=min(subs["added"] / 6, 1.0),
        observation=(
            f"Active subscriptions grew from {subs['first']} to {subs['last']} over "
            f"{subs['months']} months, with none cancelled. Recurring charges renew "
            f"by default, so they survive on inattention rather than value."
        ),
        evidence=subs,
        annual_cost=sub_cost,
    )]


def detect_impulse_buying(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Purchase bursts on consecutive days."""
    clusters = feats["impulse_clusters"]
    months = max(feats["months"], 1)
    if len(clusters) < months * 0.3:
        return []

    total = sum(c["total"] for c in clusters)
    return [_finding(
        bias="impulse_buying",
        label="Online purchases arrive in bursts",
        detected=True,
        strength=min(len(clusters) / (months * 1.5), 1.0),
        observation=(
            f"{len(clusters)} bursts of consecutive-day online shopping over "
            f"{months} months, totalling Rs {total:,.0f}. Clustered purchases are "
            f"typically one decision repeated, not several considered ones."
        ),
        evidence={"clusters": len(clusters), "largest": max(
            (c["total"] for c in clusters), default=0
        )},
        annual_cost=total / months * 12 * 0.3,   # a third is plausibly avoidable
    )]


def detect_present_focus_on_saving(feats: dict[str, Any]) -> list[dict[str, Any]]:
    """Surplus months where nothing was set aside."""
    savings = feats["savings"]
    if savings["months_with_surplus"] == 0:
        return []
    if savings["mean_capture"] >= HEALTHY_CAPTURE:
        return []

    unsaved = sum(
        m["surplus"] - m["saved"]
        for m in savings["per_month"].values() if m["surplus"] > 0
    )
    months = max(feats["months"], 1)
    return [_finding(
        bias="hyperbolic_discounting",
        label="Surplus is not being captured",
        detected=True,
        strength=1 - savings["mean_capture"] / HEALTHY_CAPTURE,
        observation=(
            f"Only {savings['mean_capture']:.1%} of surplus was saved, and "
            f"{savings['months_saved_nothing']} of "
            f"{savings['months_with_surplus']} surplus months saved nothing at all. "
            f"Money left in a spending account tends to get spent."
        ),
        evidence={
            "mean_capture": savings["mean_capture"],
            "months_saved_nothing": savings["months_saved_nothing"],
        },
        annual_cost=unsaved / months * 12,
    )]


DETECTORS = (
    detect_present_bias,
    detect_lifestyle_inflation,
    detect_subscription_creep,
    detect_impulse_buying,
    detect_present_focus_on_saving,
)


def bias_detection_advisor(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect and price behavioural patterns. Pure and deterministic."""
    if not transactions:
        return {
            "status": "no transaction history",
            "months_analysed": 0,
            "findings": [],
            "total_annual_cost": 0.0,
            "recommendations": [],
        }

    feats = F.extract_features(transactions)
    findings: list[dict[str, Any]] = []
    for detector in DETECTORS:
        findings.extend(detector(feats))

    findings.sort(key=lambda f: f["estimated_annual_cost"], reverse=True)
    total = sum(f["estimated_annual_cost"] for f in findings)

    if not findings:
        status = "No material behavioural patterns detected"
    elif len(findings) >= 5:
        status = "Multiple behavioural patterns detected"
    elif len(findings) >= 3:
        status = "Several behavioural patterns detected"
    else:
        status = "Some behavioural patterns detected"

    return {
        "status": status,
        "months_analysed": feats["months"],
        "transactions_analysed": feats["transaction_count"],
        "findings": findings,
        "biases_detected": sorted({f["bias"] for f in findings}),
        "total_annual_cost": round(total, 2),
        "features": {
            "timing": feats["timing"],
            "subscriptions": feats["subscriptions"],
            "income_trend": feats["income_trend"],
            "discretionary_trend": feats["discretionary_trend"],
            "savings": {
                "mean_capture": feats["savings"]["mean_capture"],
                "months_saved_nothing": feats["savings"]["months_saved_nothing"],
            },
        },
        "recommendations": [
            f"{f['label']} - about Rs {f['estimated_annual_cost']:,.0f}/year"
            for f in findings[:5]
        ],
    }


def bias_detection_node(
    state: FinancialState, transactions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """
    LangGraph adapter.

    Transactions come from `state["transactions"]` when a workflow loaded them,
    otherwise from the seeded demo dataset.
    """
    if transactions is None:
        transactions = state.get("transactions")
    if transactions is None:
        from ...common.synthetic import load_transactions

        transactions = load_transactions()
    return {"bias_detection_result": bias_detection_advisor(transactions)}
