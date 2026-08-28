"""
Cashflow Council -> Income Projection Agent.

Migrated verbatim from `CashFlowAdvisor/cashflow_simulator.ipynb`.

  detect_recurring       -> is this stream stable enough to treat as recurring?
                            (coefficient of variation below a threshold)
  holt_winters_forecast  -> triple-exponential smoothing, used for seasonal series
  sarimax_forecast       -> state-space model, used as the second opinion

The notebook blended the two forecasts by averaging them; `income_projection_node`
reproduces that blend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ...schemas.state import FinancialState

# Minimum observations before a seasonal model is worth fitting.
MIN_HISTORY_FOR_MODEL = 6


def detect_recurring(series: pd.Series, cv_threshold: float = 0.10, forecast_months: int = 6):
    """
    Classify a time-series as recurring if coefficient of variation < threshold.
    Returns: dict with mean, std, cv, is_recurring, forecast
    """
    mean = series.mean()
    std  = series.std()
    cv   = std / mean if mean != 0 else 99
    is_recurring = cv < cv_threshold
    # Cast out of numpy scalars: np.bool_/np.float64 are not JSON-serialisable,
    # which breaks the API and memory layers downstream.
    return {
        'mean': float(round(mean, 2)),
        'std':  float(round(std, 2)),
        'cv':   float(round(cv, 4)),
        'is_recurring': bool(is_recurring),
        'forecast': [float(round(mean, 2))] * forecast_months   # deterministic calendar
    }


def holt_winters_forecast(series: pd.Series, periods: int, seasonal_periods: int = None):
    """
    Fit Holt-Winters and return forecast array.
    Falls back to double exponential smoothing if series too short for seasonal.
    """
    n = len(series)
    try:
        if seasonal_periods and n >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                series, trend='add', seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit(optimized=True, use_brute=True)
        else:
            model = ExponentialSmoothing(
                series, trend='add', seasonal=None
            ).fit(optimized=True)
        forecast = model.forecast(periods)
        return [max(0, round(v, 2)) for v in forecast]
    except Exception as e:
        # fallback: simple mean
        return [round(series.mean(), 2)] * periods


def sarimax_forecast(series: pd.Series, periods: int,
                     order=(1,1,1), seasonal_order=(0,0,0,0),
                     exog_future=None):
    """
    SARIMAX model. Use for irregular / cyclical income streams.
    exog_future: optional array of shape (periods, n_features) for future exogenous vars.
    """
    try:
        exog_train = None
        if exog_future is not None:
            exog_train = np.tile(exog_future.mean(axis=0), (len(series), 1))

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        fc = model.forecast(periods, exog=exog_future)
        return [max(0, round(v, 2)) for v in fc]
    except Exception:
        return [round(series.mean(), 2)] * periods


# --------------------------------------------------------------------------- #
# LangGraph adapter (added during migration)
# --------------------------------------------------------------------------- #

def income_projection_node(state: FinancialState, months_ahead: int = 6) -> dict[str, Any]:
    """
    Project income forward.

    With enough history, blends Holt-Winters and SARIMAX exactly as the notebook
    did. With little or no history, falls back to a flat projection from the
    profile's stated monthly income so the agent still returns a usable answer.
    """
    profile = state["profile"]
    history = profile.income_history

    if len(history) < MIN_HISTORY_FOR_MODEL:
        flat = [round(float(profile.monthly_income), 2)] * months_ahead
        return {
            "income_projection_result": {
                "method": "flat_fallback",
                "reason": f"only {len(history)} observations, need {MIN_HISTORY_FOR_MODEL}",
                "forecast": flat,
                "recurring": None,
            }
        }

    series = pd.Series(history, dtype="float64")
    recurring = detect_recurring(series)

    hw = holt_winters_forecast(series, periods=months_ahead)
    try:
        sx = sarimax_forecast(series, periods=months_ahead)
    except Exception as exc:                      # model may fail to converge
        sx = None
        state.setdefault("errors", []).append(f"sarimax_forecast: {exc!r}")

    hw_values = hw["forecast"] if isinstance(hw, dict) else list(hw)

    if sx is None:
        blended = [round(float(v), 2) for v in hw_values]
        method = "holt_winters"
    else:
        sx_values = sx["forecast"] if isinstance(sx, dict) else list(sx)
        blended = [round((float(a) + float(b)) / 2, 2) for a, b in zip(hw_values, sx_values)]
        method = "holt_winters+sarimax_blend"

    return {
        "income_projection_result": {
            "method": method,
            "forecast": blended,
            "holt_winters": [round(float(v), 2) for v in hw_values],
            "sarimax": None if sx is None else [round(float(v), 2) for v in (
                sx["forecast"] if isinstance(sx, dict) else list(sx))],
            "recurring": recurring,
        }
    }
