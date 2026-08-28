"""
Risk Council -> Debt Trap Agent.

Migrated VERBATIM from `DebtOvercomeAdvisor/DebtOvercomeAdvisor.ipynb` via an
AST-based extractor, so the allocation arithmetic is byte-identical to the
notebook that was verified running 9/9 cells.

Two dead artefacts from the notebook were dropped during migration:
  * the first, shadowed definition of `get_effective_rate` (the second wins)
  * the `TEST {i}` harness cell, which referenced a deleted loop variable

Debt dicts use the notebook's field names -- `emi`, `last_penalty_interest`,
`outstanding_amount` -- which `ml.src.schemas.profile.Debt` mirrors exactly, so
`debt.model_dump()` can be passed straight in.

Public surface:
  rank_debts(...)        -> scenario + ranked debts
  allocate_payments(...) -> a concrete payment plan
  process_due_debts(...) -> daily cron: accrue arrears, advance overdue cycles
  debt_trap_node(state)  -> LangGraph adapter
"""

from __future__ import annotations

import copy
import datetime
import math
from datetime import timedelta
from typing import Any

from ...schemas.state import FinancialState


DEFAULT_RATES = {
    "credit_card":           36,
    "bnpl":                  24,
    "personal_loan":         15,
    "gold_loan":             12,
    "education_loan":        10,
    "car_loan":              10,
    "home_loan":              9,
    "consumer_durable_loan": 12,
    "other":                 15,
    "emi":                 15,
}


EMI_DEBTS = {
    "home_loan",
    "personal_loan",
    "education_loan",
    "car_loan",
    "gold_loan",
    "consumer_durable_loan",
    "emi"
}


def calculate_emi_arrears(
    emi,
    annual_rate,
    overdue_cycles
):
    monthly_rate = annual_rate / 1200

    arrears = 0

    for k in range(
        1,
        overdue_cycles + 1
    ):
        arrears += (
            emi
            * (1 + monthly_rate) ** k
        )

    return arrears


def get_effective_rate(debt):

    if debt.get("interest_rate") is not None:
        return debt["interest_rate"], "actual"

    return DEFAULT_RATES.get(
        debt["debt_type"],
        DEFAULT_RATES["other"]
    ), "estimated"


def calculate_available_budget(
    monthly_income,
    essential_expenses,
    emergency_fund
):
    return monthly_income - essential_expenses


def calculate_available_budget_emergency(
    monthly_income,
    essential_expenses,
    emergency_fund
):
    return monthly_income - essential_expenses + emergency_fund


def calculate_mandatory_payments(debts):

    total = 0

    for debt in debts:

        if debt["debt_type"] in EMI_DEBTS:
            total += debt.get("emi", 0)

        else:
            total += debt.get("minimum_due", 0)

    return total


def calculate_loss_if_not_paid(debt):

    rate, _ = get_effective_rate(debt)

    monthly_rate = rate / 1200

    overdue_cycles = debt.get(
        "overdue_cycles",
        0
    )

    if debt["debt_type"] in EMI_DEBTS:

        emi = debt["emi"]

        return (
            emi
            * ((1 + monthly_rate) ** (overdue_cycles + 1))
            - emi
        )

    minimum_due = debt["minimum_due"]

    return (
        debt["outstanding_amount"] * monthly_rate 
    )


def get_debt_bucket(debt):

    if debt["debt_type"] in EMI_DEBTS:
        return "emi"

    return "revolving"


def rank_debts(
    debts,
    monthly_income,
    essential_expenses,
    emergency_fund
):

    available_budget = calculate_available_budget(
        monthly_income,
        essential_expenses,
        emergency_fund
    )

    available_budget_emergency = calculate_available_budget_emergency(
        monthly_income,
        essential_expenses,
        emergency_fund
    )

    mandatory_payments = calculate_mandatory_payments(
        debts
    )

    ranked = []

    for debt in debts:

        loss = calculate_loss_if_not_paid(
            debt
        )

        ranked.append({
            **debt,
            "loss_if_not_paid": round(
                loss,
                2
            )
        })

    #
    # CASE A
    #
    if available_budget >= mandatory_payments:

        ranked.sort(
            key=lambda d: d["interest_rate"],
            reverse=True
        )

        return {
            "scenario": "income_sufficient",
            "ranked_debts": ranked
        }

    #
    # CASE B
    #
    if (
        available_budget
        + emergency_fund
        >= mandatory_payments
    ):

        ranked.sort(
            key=lambda d: d["interest_rate"],
            reverse=True
        )

        return {
            "scenario": "use_emergency_fund",
            "ranked_debts": ranked
        }

    #
    # CASE C
    #
    bucket_priority = {
        "emi": 0,
        "revolving": 1
    }

    ranked.sort(
        key=lambda d: (
            bucket_priority[
                get_debt_bucket(d)
            ],
            d["outstanding_amount"],
        )
    )

    return {
        "scenario": "cannot_cover_mandatory",
        "ranked_debts": ranked
    }


def allocate_payments(
    monthly_income,
    essential_expenses,
    emergency_fund,
    scenario,
    debts
):

    proposed_solution = {
        "remaining_income": 0,
        "remaining_emergency_fund": emergency_fund,
        "emergency_fund_used": 0,
        "debts": copy.deepcopy(debts)
    }

    available_income = (
        monthly_income
        - essential_expenses
    )

    mandatory_payments = (
        calculate_mandatory_payments(
            debts
        )
    )

    # --------------------------------------------------
    # Determine usable cash
    # --------------------------------------------------

    if scenario == "income_sufficient":

        emergency_used = 0

        usable_cash = available_income

    elif scenario == "use_emergency_fund":

        shortfall = max(
            mandatory_payments
            - available_income,
            0
        )

        emergency_used = min(
            shortfall,
            emergency_fund
        )

        usable_cash = (
            available_income
            + emergency_used
        )

        emergency_fund -= emergency_used

    else:  # cannot_cover_mandatory

        emergency_used = emergency_fund

        usable_cash = (
            available_income
            + emergency_fund
        )

        emergency_fund = 0

    remaining_cash = usable_cash

    # --------------------------------------------------
    # Initialize fields
    # --------------------------------------------------

    for debt in proposed_solution["debts"]:

        payment = (
            debt.get("emi")
            or debt.get("minimum_due")
            or 0
        )

        debt["required_payment"] = payment

        debt["mandatory_paid"] = 0

        debt["extra_payment"] = 0

        debt["total_paid"] = 0

        debt["shortfall"] = payment

    # --------------------------------------------------
    # STEP 1
    # Mandatory payments
    # --------------------------------------------------

    for debt in proposed_solution["debts"]:

        payment = debt["required_payment"]

        if remaining_cash <= 0:
            break

        allocated = min(
            payment,
            remaining_cash
        )

        debt["mandatory_paid"] = allocated

        debt["shortfall"] = (
            payment - allocated
        )

        debt["outstanding_amount"] = max(
            debt["outstanding_amount"]
            - allocated,
            0
        )

        # # reduce overdue count by one cycle
        # if (
        #     allocated >= payment
        #     and debt.get(
        #         "overdue_cycles"
        #     ) is not None
        # ):
        #     debt["overdue_cycles"] = max(
        #         debt.get(
        #             "overdue_cycles",
        #             0
        #         ) - 1,
        #         0
        #     )

        # remaining_cash -= allocated

        # reduce overdue count by one cycle
        if (
            allocated >= payment
            and debt.get(
                "overdue_cycles"
            ) is not None
        ):
            debt["overdue_cycles"] = 0

        remaining_cash -= allocated

        if allocated < payment:

            debt["shortfall"] = (
                payment - allocated
            )

            monthly_rate = (
                debt.get(
                    "interest_rate",
                    0
                ) / 100
            ) / 12

            debt["outstanding_amount"] *= (
                1 + monthly_rate
            )

            if (
                debt.get(
                    "overdue_cycles"
                ) is not None
            ):
                debt["overdue_cycles"] += 1
        else:

            if (
                debt.get(
                    "overdue_cycles"
                ) is not None
            ):
                debt["overdue_cycles"] = 0

        
    # --------------------------------------------------
    # STEP 2
    # Avalanche surplus allocation
    # Debts already arrive ranked
    # --------------------------------------------------
# --------------------------------------------------
# STEP 2
# Avalanche surplus allocation
# --------------------------------------------------

    if remaining_cash > 0:

        for debt in proposed_solution["debts"]:

            if remaining_cash <= 0:
                break

            outstanding = (
                debt["outstanding_amount"]
            )

            if outstanding <= 0:
                continue

            extra = min(
                remaining_cash,
                outstanding
            )

            debt["extra_payment"] = extra

            debt["outstanding_amount"] -= extra

            if (
                debt["outstanding_amount"]
                <= 0
            ):
                if (
                    debt.get(
                        "overdue_cycles"
                    ) is not None
                ):
                    debt["overdue_cycles"] = 0

            remaining_cash -= extra
    # --------------------------------------------------
    # Final totals
    # --------------------------------------------------

    for debt in proposed_solution["debts"]:

            debt["total_paid"] = (
                    debt["mandatory_paid"]
                    + debt["extra_payment"]
                )

            proposed_solution[
                "mandatory_payments_met"
            ] = all(
                debt["mandatory_paid"]
                >= debt["required_payment"]
                for debt in proposed_solution[
                    "debts"
                ]
            )

            proposed_solution[
                "remaining_income"
            ] = remaining_cash

            proposed_solution[
                "remaining_emergency_fund"
            ] = emergency_fund

            proposed_solution[
                "emergency_fund_used"
            ] = emergency_used

            proposed_solution[
                "mandatory_payments"
            ] = mandatory_payments

    return proposed_solution


def process_due_debts(debts):
    """
    Daily cron job (12:01 AM).

    Logic:
    1. Check if yesterday was the due date.
    2. Determine required payment.
    3. Determine shortfall.
    4. If shortfall exists:
         - Increment overdue cycle
         - Add shortfall to outstanding
         - Add monthly interest ONLY on shortfall
    5. Do not apply interest on the entire outstanding balance.
    """

    yesterday_day = (
        datetime.now() - timedelta(days=1)
    ).day

    for debt in debts:

        due_day = int(debt["date_of_payment"])

        if due_day != yesterday_day:
            continue

        required_payment = max(
            debt.get("minimum_due", 0),
            debt.get("emi", 0)
        )

        paid_amount = debt.get(
            "paid_this_cycle",
            0
        )

        shortfall = max(
            0,
            required_payment - paid_amount
        )

        if shortfall == 0:
            continue

        debt["overdue_cycles"] = (
            debt.get("overdue_cycles", 0) + 1
        )

        monthly_rate = (
            debt["interest_rate"]
            / 12
            / 100
        )

        penalty_interest = (
            shortfall * monthly_rate
        )

        debt["outstanding_amount"] += (
            shortfall +
            penalty_interest
        )

        debt["last_serviced"] = datetime.now().strftime(
            "%Y-%m-%d"
        )

        debt["last_shortfall"] = round(
            shortfall,
            2
        )

        debt["last_penalty_interest"] = round(
            penalty_interest,
            2
        )

    return debts


# --------------------------------------------------------------------------- #
# LangGraph adapter (added during migration; not part of the original notebook)
# --------------------------------------------------------------------------- #

def debt_trap_node(state: FinancialState) -> dict[str, Any]:
    """Rank the user's debts and produce an allocation plan."""
    profile = state["profile"]
    debts = [d.model_dump() for d in profile.debts]

    if not debts:
        return {
            "debt_trap_result": {
                "scenario": "no_debt",
                "ranked_debts": [],
                "allocation": None,
                "total_debt": 0.0,
                "debt_to_income": 0.0,
            }
        }

    ranked = rank_debts(
        debts=debts,
        monthly_income=profile.monthly_income,
        essential_expenses=profile.essential_expenses,
        emergency_fund=profile.existing_emergency_fund,
    )

    allocation = allocate_payments(
        monthly_income=profile.monthly_income,
        essential_expenses=profile.essential_expenses,
        emergency_fund=profile.existing_emergency_fund,
        scenario=ranked["scenario"],
        debts=ranked["ranked_debts"],
    )

    return {
        "debt_trap_result": {
            "scenario": ranked["scenario"],
            "ranked_debts": ranked["ranked_debts"],
            "allocation": allocation,
            "total_debt": profile.total_debt,
            "debt_to_income": profile.debt_to_income,
        }
    }
