"""
Decision Layer -> Monte Carlo Engine.

The engine is seeded, so every assertion here is exact rather than statistical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.decision.montecarlo import statistical_estimator


@pytest.fixture
def series() -> pd.Series:
    return pd.Series([44_000, 46_000, 45_000, 47_000, 52_000, 44_000], dtype="float64")


def test_returns_mean_p10_p90_of_requested_length(series):
    out = statistical_estimator(series, periods=6)
    assert set(out) == {"mean", "p10", "p90"}
    assert all(len(out[k]) == 6 for k in out)


def test_bands_are_ordered_p10_below_mean_below_p90(series):
    out = statistical_estimator(series, periods=6)
    for lo, mid, hi in zip(out["p10"], out["mean"], out["p90"]):
        assert lo <= mid <= hi


def test_is_deterministic_for_a_fixed_seed(series):
    a = statistical_estimator(series, periods=4, seed=42)
    b = statistical_estimator(series, periods=4, seed=42)
    assert a == b


def test_different_seeds_give_different_paths(series):
    a = statistical_estimator(series, periods=4, seed=1)
    b = statistical_estimator(series, periods=4, seed=2)
    assert a != b


def test_mean_forecast_tracks_the_input_mean(series):
    out = statistical_estimator(series, periods=12, n_simulations=5_000)
    assert np.mean(out["mean"]) == pytest.approx(series.mean(), rel=0.05)


def test_never_projects_negative_spending():
    """Spending is clipped at zero even when sigma dwarfs the mean."""
    volatile = pd.Series([100.0, 20_000.0, 50.0, 30_000.0], dtype="float64")
    out = statistical_estimator(volatile, periods=6)
    assert all(v >= 0 for v in out["mean"])
    assert all(v >= 0 for v in out["p10"])
    assert all(v >= 0 for v in out["p90"])


def test_zero_variance_series_collapses_to_a_constant():
    flat = pd.Series([50_000.0] * 8, dtype="float64")
    out = statistical_estimator(flat, periods=3)
    assert out["mean"] == pytest.approx([50_000.0] * 3, abs=1.0)
    assert out["p10"] == pytest.approx(out["p90"], abs=1.0)


def test_more_simulations_narrows_the_mean_estimate(series):
    """The sample mean should sit closer to the true mean with more draws."""
    true_mean = series.mean()
    few = statistical_estimator(series, periods=1, n_simulations=50, seed=7)
    many = statistical_estimator(series, periods=1, n_simulations=20_000, seed=7)
    assert abs(many["mean"][0] - true_mean) <= abs(few["mean"][0] - true_mean)
