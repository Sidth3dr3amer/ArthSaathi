"""
Behavioral Council -> shared feature extraction.

Asserted against the seeded dataset, whose planted signals are ground truth.
"""

from __future__ import annotations

import pytest

from ml.src.common.synthetic import generate_transactions, load_transactions
from ml.src.councils.behavioral import features as F


@pytest.fixture(scope="module")
def txns():
    return load_transactions()


@pytest.fixture(scope="module")
def feats(txns):
    return F.extract_features(txns)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def test_seeded_dataset_is_present_and_substantial(txns):
    assert len(txns) > 500
    assert {"date", "amount", "category", "direction"} <= set(txns[0])


def test_generator_is_deterministic():
    a = generate_transactions(months=6, seed=7)["transactions"]
    b = generate_transactions(months=6, seed=7)["transactions"]
    assert a == b


def test_different_seeds_differ():
    a = generate_transactions(months=6, seed=1)["transactions"]
    b = generate_transactions(months=6, seed=2)["transactions"]
    assert a != b


def test_generator_declares_its_planted_signals():
    meta = generate_transactions(months=3)["meta"]
    assert meta["synthetic"] is True
    assert "month_end_splurge" in meta["planted_signals"]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_months_are_counted(feats):
    assert feats["months"] == 24


def test_income_and_spend_are_separated(feats):
    assert all(v > 0 for v in feats["monthly_income"].values())
    assert all(v > 0 for v in feats["monthly_spend"].values())


def test_category_totals_are_ranked(feats):
    values = list(feats["category_totals"].values())
    assert values == sorted(values, reverse=True)


def test_empty_history_yields_empty_features():
    out = F.extract_features([])
    assert out["months"] == 0
    assert out["category_totals"] == {}
    assert out["recurring"] == []


# --------------------------------------------------------------------------- #
# Planted signal: month-end and payday spikes
# --------------------------------------------------------------------------- #

def test_per_category_timing_recovers_the_planted_spike(feats):
    """
    The spike was planted in dining and entertainment only. A blended average
    dilutes it below the detection threshold, which is why the agent reads the
    per-category view.
    """
    by_cat = feats["timing"]["by_category"]
    assert by_cat["dining"]["month_end_ratio"] > 1.5
    assert by_cat["entertainment"]["month_end_ratio"] > 1.5


def test_untouched_categories_show_no_month_end_effect(feats):
    by_cat = feats["timing"]["by_category"]
    assert by_cat["offline_retail"]["month_end_ratio"] < 1.3
    assert by_cat["travel"]["month_end_ratio"] < 1.3


def test_salary_day_spike_is_detected(feats):
    assert feats["timing"]["by_category"]["dining"]["salary_day_ratio"] > 1.4


def test_timing_reports_sample_sizes(feats):
    for stats in feats["timing"]["by_category"].values():
        assert stats["sample_size"] == sum(
            stats["counts"][k] for k in ("early", "middle", "late")
        )


# --------------------------------------------------------------------------- #
# Planted signal: subscription creep
# --------------------------------------------------------------------------- #

def test_subscription_growth_is_detected(feats):
    subs = feats["subscriptions"]
    assert subs["last"] > subs["first"]
    assert subs["added"] >= 3


# --------------------------------------------------------------------------- #
# Planted signal: lifestyle inflation
# --------------------------------------------------------------------------- #

def test_discretionary_grows_faster_than_income(feats):
    assert feats["discretionary_trend"] > feats["income_trend"] > 0


def test_trend_needs_enough_history():
    assert F.trend({"2026-01": 100.0, "2026-02": 200.0}) == 0.0


def test_trend_compares_thirds_not_endpoints():
    """One unusual final month must not masquerade as a trend."""
    flat_with_spike = {f"2026-{i:02d}": 100.0 for i in range(1, 12)}
    flat_with_spike["2026-12"] = 900.0
    assert F.trend(flat_with_spike) < 3.0


# --------------------------------------------------------------------------- #
# Planted signal: impulse clusters and thin saving
# --------------------------------------------------------------------------- #

def test_impulse_clusters_are_consecutive_day_runs(feats):
    clusters = feats["impulse_clusters"]
    assert clusters
    assert all(c["days"] >= 2 for c in clusters)


def test_savings_capture_is_low_as_planted(feats):
    savings = feats["savings"]
    assert savings["mean_capture"] < 0.15
    assert savings["months_saved_nothing"] > 5


# --------------------------------------------------------------------------- #
# Recurring merchants
# --------------------------------------------------------------------------- #

def test_recurring_requires_three_distinct_months():
    txns = [
        {"date": "2026-01-05", "amount": 500, "category": "dining",
         "merchant": "Once", "direction": "debit"},
        {"date": "2026-01-06", "amount": 500, "category": "dining",
         "merchant": "Once", "direction": "debit"},
    ]
    assert F.recurring_merchants(txns) == []


def test_recurring_is_ranked_by_annual_cost(feats):
    costs = [m["annual_cost"] for m in feats["recurring"]]
    assert costs == sorted(costs, reverse=True)
