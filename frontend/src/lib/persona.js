/*
  The demo persona.

  Deliberately the deck's own slide-9 user rather than a comfortable one:
  a Nashik smallholder on Rs 35,000 a month, no health cover, three
  dependants, and a 42% card. That is who this product is for, and it is
  the profile that makes every council have something worth saying.
*/

export const RAHUL = {
  user_id: "rahul-patil",
  name: "Rahul Patil",
  age: 42,
  job_type: "business",
  occupation: "farmer",
  state: "Maharashtra",
  residence: "rural",
  gender: "male",
  social_category: "obc",
  land_holding_ha: 1.2,

  monthly_income: 35000,
  essential_expenses: 22000,
  current_balance: 18000,
  existing_emergency_fund: 12000,
  retirement_corpus: 40000,
  monthly_investment: 0,
  annual_household_income: 420000,

  dependents: 3,
  has_health_insurance: false,
  has_life_insurance: false,
  has_term_cover: false,
  has_bank_account: true,
  aadhaar_linked: true,
  is_income_tax_payer: false,
  is_govt_employee: false,

  monthly_spend: {
    groceries: 7000,
    utility_bills: 1800,
    fuel: 2200,
    dining: 1500,
    healthcare: 1200,
    education: 3000,
    others: 1500,
  },

  max_annual_fee: 500,
  prefer_cashback: true,
  travel_frequency: "none",

  debts: [
    {
      name: "Gold loan",
      debt_type: "gold_loan",
      outstanding_amount: 90000,
      interest_rate: 12.0,
      emi: 4200,
    },
    {
      name: "Credit card",
      debt_type: "credit_card",
      outstanding_amount: 48000,
      interest_rate: 42.0,
      minimum_due: 2400,
    },
  ],

  goals: [
    {
      name: "Tractor down payment",
      target_amount: 250000,
      current_amount: 20000,
      target_months: 36,
      priority: "high",
    },
    {
      name: "Daughter's education",
      target_amount: 600000,
      current_amount: 45000,
      target_months: 96,
      priority: "high",
    },
  ],
};

/** A blank profile for the onboarding screen — nothing known yet. */
export const NEW_USER = { user_id: "new-user", name: "" };

export const PRESETS = [
  { label: "Full review", q: "give me a full financial review" },
  { label: "Schemes", q: "am I eligible for any government schemes?" },
  { label: "Salary day", q: "my salary just got credited, what should I do?" },
  { label: "Goal", q: "I want to save for a tractor down payment" },
  { label: "Card", q: "which credit card should I get?" },
  { label: "Life event", q: "my daughter is getting married next year" },
  { label: "Forecast", q: "how much will I have in 6 months?" },
  { label: "Safety", q: "do I have enough emergency fund and insurance?" },
];
