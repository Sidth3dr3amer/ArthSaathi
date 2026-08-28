"""Cashflow Council -> Income Projection Agent."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.src.councils.cashflow.income_projection import (
    MIN_HISTORY_FOR_MODEL,
    detect_recurring,
    holt_winters_forecast,
    income_projection_node,
    sarimax_forecast,
)
from ml.src.schemas.profile import UserProfile
from ml.src.schemas.state import new_state


# --------------------------------------------------------------------------- #
# Recurrence detection
# --------------------------------------------------------------------------- #

def test_stable_salary_is_detected_as_recurring():
    out = detect_recurring(pd.Series([50_000, 50_100, 49_900, 50_050], dtype="float64"))
    assert out["is_recurring"] is True
    assert out["cv"] < 0.10


def test_volatile_gig_income_is_not_recurring():
    out = detect_recurring(pd.Series([12_000, 45_000, 8_000, 60_000], dtype="float64"))
    assert out["is_recurring"] is False
    assert out["cv"] > 0.10


def test_all_zero_series_does_not_divide_by_zero():
    out = detect_recurring(pd.Series([0.0, 0.0, 0.0], dtype="float64"))
    assert out["cv"] == 99
    assert out["is_recurring"] is False


def test_recurring_forecast_length_follows_forecast_months():
    out = detect_recurring(pd.Series([50_000.0] * 6), forecast_months=9)
    assert len(out["forecast"]) == 9


# --------------------------------------------------------------------------- #
# Forecasters
# --------------------------------------------------------------------------- #

@pytest.fixture
def trending() -> pd.Series:
    return pd.Series(
        [58_000, 60_000, 59_000, 61_000, 63_000, 62_000,
         64_000, 66_000, 65_000, 67_000, 69_000, 68_000],
        dtype="float64",
    )


def test_holt_winters_returns_requested_horizon(trending):
    assert len(holt_winters_forecast(trending, periods=6)) == 6


def test_sarimax_returns_requested_horizon(trending):
    assert len(sarimax_forecast(trending, periods=6)) == 6


@pytest.mark.parametrize("fn", [holt_winters_forecast, sarimax_forecast])
def test_forecasters_never_return_negative_income(fn, trending):
    assert all(v >= 0 for v in fn(trending, periods=6))


@pytest.mark.parametrize("fn", [holt_winters_forecast, sarimax_forecast])
def test_forecasters_fall_back_to_the_mean_on_unfittable_input(fn):
    """Both wrap their model in try/except and degrade to a flat mean."""
    tiny = pd.Series([1_000.0, 2_000.0], dtype="float64")
    out = fn(tiny, periods=4)
    assert len(out) == 4
    assert all(v >= 0 for v in out)


def test_holt_winters_follows_an_upward_trend(trending):
    out = holt_winters_forecast(trending, periods=6)
    assert out[-1] > trending.iloc[-1] * 0.9


# --------------------------------------------------------------------------- #
# Node adapter
# --------------------------------------------------------------------------- #

def test_node_blends_both_models_when_history_is_sufficient(history_profile):
    patch = income_projection_node(new_state(history_profile), months_ahead=6)
    assert set(patch) == {"income_projection_result"}

    result = patch["income_projection_result"]
    assert result["method"] == "holt_winters+sarimax_blend"
    assert len(result["forecast"]) == 6
    assert result["recurring"]["is_recurring"] is True

    # the blend must sit between its two inputs
    for blended, hw, sx in zip(result["forecast"], result["holt_winters"], result["sarimax"]):
        assert min(hw, sx) - 0.01 <= blended <= max(hw, sx) + 0.01


def test_node_falls_back_to_flat_projection_without_history(salaried_profile):
    result = income_projection_node(new_state(salaried_profile), months_ahead=4)[
        "income_projection_result"
    ]
    assert result["method"] == "flat_fallback"
    assert result["forecast"] == [90_000.0] * 4
    assert result["recurring"] is None


def test_node_fallback_boundary_is_min_history_for_model():
    just_under = UserProfile(
        user_id="t", monthly_income=50_000,
        income_history=[50_000.0] * (MIN_HISTORY_FOR_MODEL - 1),
    )
    result = income_projection_node(new_state(just_under))["income_projection_result"]
    assert result["method"] == "flat_fallback"


def test_node_horizon_is_respected(history_profile):
    result = income_projection_node(new_state(history_profile), months_ahead=12)[
        "income_projection_result"
    ]
    assert len(result["forecast"]) == 12
