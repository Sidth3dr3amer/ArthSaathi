"""
Durable facts about a user.

`UserProfile` is the single source of truth that every council agent reads from.
It unifies the input shapes that were previously scattered across notebooks:

  * EmergencyFundAdvisor  -> income, expenses, existing_emergency_fund, job_type,
                             dependents, has_health_insurance
  * DebtOvercomeAdvisor   -> debts[], monthly_income, essential_expenses, emergency_fund
  * credit_card engine    -> age, employment_type, monthly_spend{}, lifestyle flags
  * cashflow_simulator    -> current_balance, income/expense history
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

JobType = Literal["salaried", "govt", "freelancer", "business", "student", "unsalaried"]

# Values must match DEFAULT_RATES / EMI_DEBTS in ml/src/councils/risk/debt_trap.py.
DebtType = Literal[
    "credit_card", "bnpl", "personal_loan", "gold_loan", "education_loan",
    "car_loan", "home_loan", "consumer_durable_loan", "emi", "other",
]


class Debt(BaseModel):
    """
    One liability.

    Field names match `DebtOvercomeAdvisor` exactly — notably `emi` (not
    `emi_amount`) and `last_penalty_interest` — so profiles can be handed to the
    migrated functions as plain dicts via `.model_dump()` with no translation.
    """

    name: str
    debt_type: DebtType
    outstanding_amount: float = Field(ge=0)
    interest_rate: float | None = Field(
        default=None, ge=0,
        description="Annual nominal rate, percent. None falls back to DEFAULT_RATES.",
    )
    minimum_due: float = Field(default=0, ge=0)
    emi: float = Field(default=0, ge=0)
    overdue_cycles: int = Field(default=0, ge=0)
    last_shortfall: float = Field(default=0, ge=0)
    last_penalty_interest: float = Field(default=0, ge=0)
    last_serviced: str | None = None
    due_day: int | None = Field(default=None, ge=1, le=31)


class Goal(BaseModel):
    """A financial goal the user is saving toward."""

    name: str
    target_amount: float = Field(gt=0)
    current_amount: float = Field(default=0, ge=0)
    target_months: int = Field(gt=0)
    priority: Literal["low", "medium", "high"] = "medium"

    @computed_field
    @property
    def remaining(self) -> float:
        return max(0.0, self.target_amount - self.current_amount)


class UserProfile(BaseModel):
    """Everything the councils need to reason about one person."""

    user_id: str = "demo-user"
    name: str = "Demo User"

    # --- Identity / eligibility ---
    age: int = Field(default=30, ge=16, le=100)
    job_type: JobType = "salaried"
    dependents: int = Field(default=0, ge=0)
    state: str | None = None
    occupation: str | None = None

    # --- Government-scheme eligibility signals (Benefits Council) ---
    gender: Literal["male", "female", "other"] | None = None
    social_category: Literal["general", "obc", "sc", "st"] | None = None
    residence: Literal["rural", "urban"] | None = None
    land_holding_ha: float | None = Field(
        default=None, ge=0, description="Agricultural land in hectares"
    )
    is_income_tax_payer: bool = False
    is_govt_employee: bool = False
    has_bank_account: bool = True
    aadhaar_linked: bool = True
    annual_household_income: float | None = Field(default=None, ge=0)

    # --- Cash position ---
    monthly_income: float = Field(default=0, ge=0)
    essential_expenses: float = Field(default=0, ge=0)
    current_balance: float = Field(default=0)
    existing_emergency_fund: float = Field(default=0, ge=0)

    # --- Liabilities & goals ---
    debts: list[Debt] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)

    # --- Long-term assets (Growth Council) ---
    #: Retirement savings only -- EPF/NPS/PPF/equity earmarked for retirement.
    #: Deliberately separate from `existing_emergency_fund`: runway is not a
    #: retirement asset and counting it as one overstates readiness.
    retirement_corpus: float = Field(default=0, ge=0)
    monthly_investment: float = Field(default=0, ge=0)
    current_allocation: dict[str, float] = Field(default_factory=dict)

    # --- Protection ---
    has_health_insurance: bool = True
    has_life_insurance: bool = False
    has_term_cover: bool = False

    # --- Spend breakdown (credit-card + behavioural councils) ---
    monthly_spend: dict[str, float] = Field(default_factory=dict)

    # --- Time series (cashflow council: forecasting + simulation) ---
    # Oldest-first monthly totals. The Holt-Winters and SARIMAX forecasters need
    # a minimum history length; the cashflow node falls back to a flat estimate
    # from monthly_income / essential_expenses when these are empty.
    income_history: list[float] = Field(default_factory=list)
    expense_history: list[float] = Field(default_factory=list)

    # --- Preferences ---
    max_annual_fee: float = Field(default=0, ge=0)
    prefer_cashback: bool = True
    prefer_travel_perks: bool = False
    travel_frequency: Literal["none", "occasional", "frequent"] = "none"
    lifestyle_flags: dict[str, bool] = Field(default_factory=dict)

    @computed_field
    @property
    def monthly_surplus(self) -> float:
        """Income minus essential expenses and mandatory debt payments."""
        mandatory = sum(max(d.minimum_due, d.emi) for d in self.debts)
        return self.monthly_income - self.essential_expenses - mandatory

    @computed_field
    @property
    def total_debt(self) -> float:
        return sum(d.outstanding_amount for d in self.debts)

    @computed_field
    @property
    def total_monthly_spend(self) -> float:
        return sum(self.monthly_spend.values())

    @computed_field
    @property
    def debt_to_income(self) -> float:
        """Monthly debt servicing as a fraction of monthly income."""
        if self.monthly_income <= 0:
            return 0.0
        mandatory = sum(max(d.minimum_due, d.emi) for d in self.debts)
        return round(mandatory / self.monthly_income, 4)
