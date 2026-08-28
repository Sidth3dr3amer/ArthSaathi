"""
Cashflow Council -> Cashflow Stability Agent.

Migrated verbatim from `CashFlowAdvisor/cashflow_simulator.ipynb`.

  cashflow_simulator -> month-by-month balance projection under three scenarios
                        (base / optimistic / pessimistic)
  risk_engine        -> scores that projection 0-100 and raises human-readable flags

One signature change during migration: `today` was a module-level global in the
notebook and is now an injectable parameter defaulting to `datetime.today()`.
Behaviour is identical; the parameter exists so tests are not date-dependent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ...decision.montecarlo import statistical_estimator
from ...schemas.state import FinancialState


def cashflow_simulator(current_balance, income_fc, expense_fc_base,
                       expense_fc_p10, expense_fc_p90, months_ahead, today=None):
    """
    Simulate month-by-month cashflow under three scenarios.
    Returns DataFrame with columns: month, income, expense_{base/opt/pess},
    balance_{base/opt/pess}, net_{base/opt/pess}
    """
    today = today or datetime.today()

    records = []
    bal_base = bal_opt = bal_pess = current_balance

    for i in range(months_ahead):
        inc  = income_fc[i]
        exp_base = expense_fc_base[i]
        exp_opt  = expense_fc_p10[i]
        exp_pess = expense_fc_p90[i]

        net_base = inc - exp_base
        net_opt  = inc - exp_opt
        net_pess = inc - exp_pess

        bal_base += net_base
        bal_opt  += net_opt
        bal_pess += net_pess

        future_month = today + relativedelta(months=i+1)
        records.append({
            'month':       future_month.strftime('%b %Y'),
            'income':      round(inc, 0),
            'exp_base':    round(exp_base, 0),
            'exp_opt':     round(exp_opt, 0),
            'exp_pess':    round(exp_pess, 0),
            'net_base':    round(net_base, 0),
            'net_opt':     round(net_opt, 0),
            'net_pess':    round(net_pess, 0),
            'bal_base':    round(bal_base, 0),
            'bal_opt':     round(bal_opt, 0),
            'bal_pess':    round(bal_pess, 0),
        })

    return pd.DataFrame(records).set_index('month')


def risk_engine(sim_df, current_balance, monthly_expenses_avg,
                dependents, goal_text, external_factors_text,
                income_values=None):
    """
    Score and flag risks from the cashflow simulation.
    Returns a dict of risk flags and scores.
    """
    risks = []
    score = 0   # 0 = low risk, 100 = high risk

    # 1. Negative balance risk
    if (sim_df['bal_base'] < 0).any():
        risks.append('🔴 Negative balance projected in base scenario')
        score += 30
    if (sim_df['bal_pess'] < 0).any():
        risks.append('🟠 Negative balance possible in pessimistic scenario')
        score += 15

    # 2. Emergency fund check (3 months expenses)
    emergency_target = monthly_expenses_avg * 3
    min_balance = sim_df['bal_base'].min()
    if min_balance < emergency_target:
        shortfall = emergency_target - min_balance
        risks.append(f'🟡 Emergency fund short by ₹{shortfall:,.0f} (need {int(emergency_target):,})')
        score += 20

    # 3. Income volatility
    # `income_values` was a notebook global; it now defaults to the simulated
    # income column so the function is self-contained.
    if income_values is None:
        income_values = list(sim_df['income'])
    _inc = pd.Series(income_values, dtype='float64')
    _inc_mean = _inc.mean()
    income_cv = (_inc.std() / _inc_mean) if _inc_mean else 0.0
    if income_cv > 0.15:
        risks.append(f'🟡 High income volatility (CV={income_cv:.2f})')
        score += 10

    # 4. Dependents
    if dependents >= 2:
        risks.append(f'🟡 {dependents} dependents increase financial exposure')
        score += 10

    # 5. External factors keyword scan
    high_risk_kw = ['inflation', 'job loss', 'medical', 'loan', 'emi', 'debt']
    for kw in high_risk_kw:
        if kw in external_factors_text.lower():
            risks.append(f'🟠 External risk keyword detected: "{kw}"')
            score += 5

    # 6. Savings rate
    avg_net = sim_df['net_base'].mean()
    avg_inc = sim_df['income'].mean()
    savings_rate = avg_net / avg_inc if avg_inc > 0 else 0
    if savings_rate < 0.10:
        risks.append(f'🔴 Savings rate very low ({savings_rate:.1%})')
        score += 20
    elif savings_rate < 0.20:
        risks.append(f'🟡 Savings rate below recommended 20% ({savings_rate:.1%})')
        score += 10

    # Overall rating
    score = min(score, 100)
    if score < 25:   rating = '🟢 LOW'
    elif score < 55: rating = '🟡 MODERATE'
    elif score < 80: rating = '🟠 HIGH'
    else:            rating = '🔴 CRITICAL'

    return {
        'score': score,
        'rating': rating,
        'flags': risks,
        'savings_rate': savings_rate,
        'emergency_target': emergency_target,
        'min_projected_balance': min_balance
    }


# --------------------------------------------------------------------------- #
# LangGraph adapter (added during migration)
# --------------------------------------------------------------------------- #

def stability_node(state: FinancialState, months_ahead: int = 6) -> dict[str, Any]:
    """
    Project the user's balance forward and score the resulting risk.

    Uses the income forecast produced upstream by `income_projection_node` when
    present, so the two cashflow agents compose. Expenses are modelled with the
    Monte Carlo estimator when history exists, otherwise held flat.
    """
    profile = state["profile"]

    upstream = state.get("income_projection_result") or {}
    income_fc = upstream.get("forecast") or [float(profile.monthly_income)] * months_ahead
    income_fc = [float(v) for v in income_fc][:months_ahead]
    while len(income_fc) < months_ahead:
        income_fc.append(float(profile.monthly_income))

    if len(profile.expense_history) >= 3:
        series = pd.Series(profile.expense_history, dtype="float64")
        mc = statistical_estimator(series, periods=months_ahead)
        exp_base, exp_opt, exp_pess = mc["mean"], mc["p10"], mc["p90"]
    else:
        flat = float(profile.essential_expenses)
        exp_base = [flat] * months_ahead
        exp_opt = [flat * 0.9] * months_ahead
        exp_pess = [flat * 1.15] * months_ahead

    sim_df = cashflow_simulator(
        current_balance=profile.current_balance,
        income_fc=income_fc,
        expense_fc_base=exp_base,
        expense_fc_p10=exp_opt,
        expense_fc_p90=exp_pess,
        months_ahead=months_ahead,
    )

    goal_text = "; ".join(g.name for g in profile.goals) or "no stated goals"
    risk = risk_engine(
        sim_df=sim_df,
        current_balance=profile.current_balance,
        monthly_expenses_avg=float(np.mean(exp_base)),
        dependents=profile.dependents,
        goal_text=goal_text,
        external_factors_text=state.get("query", "") or "none stated",
    )

    return {
        "stability_result": {
            "months_ahead": months_ahead,
            "projection": sim_df.reset_index().to_dict(orient="records"),
            "risk": risk,
            "income_forecast": income_fc,
            "expense_forecast_base": exp_base,
        },
        "simulation_result": {
            "engine": "monte_carlo" if len(profile.expense_history) >= 3 else "flat",
            "months_ahead": months_ahead,
        },
    }
