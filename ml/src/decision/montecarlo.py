"""
Decision Layer -> Monte Carlo Engine.

Migrated verbatim from `CashFlowAdvisor/cashflow_simulator.ipynb`. This is the
"Monte Carlo Engine" node in the deck's orchestration diagram: it draws
`n_simulations` paths for irregular/discretionary spend and returns the mean
plus optimistic (p10) and pessimistic (p90) bands per period.

Seeded by default, so results are reproducible and directly unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def statistical_estimator(series: pd.Series, periods: int,
                           n_simulations: int = 1000, seed: int = 42):
    """
    Monte Carlo estimator for discretionary / irregular spending.
    Returns: mean forecast, p10 (optimistic), p90 (pessimistic) per period.
    """
    rng  = np.random.default_rng(seed)
    mu   = series.mean()
    sigma = series.std()

    sims = rng.normal(loc=mu, scale=sigma, size=(n_simulations, periods))
    sims = np.clip(sims, 0, None)   # no negative spending

    return {
        'mean': [round(v, 2) for v in sims.mean(axis=0)],
        'p10':  [round(v, 2) for v in np.percentile(sims, 10, axis=0)],
        'p90':  [round(v, 2) for v in np.percentile(sims, 90, axis=0)],
    }
