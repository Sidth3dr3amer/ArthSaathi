Overall flow

I would build a **Credit Card Intelligence Agent System**.

# Tier 0: Knowledge Base Builder (Daily Cron)

```text
Google Discovery Agent
        ↓
Document Collector Agent
        ↓
Document Parser Agent
        ↓
Feature Extraction Agent
        ↓
Validation Agent
        ↓
Card Knowledge Base
```

---

## 1. Google Discovery Agent

Input:

```text
HDFC
Regalia Gold
```

Generates:

```text
site:hdfc.bank.in "Regalia Gold" filetype:pdf
site:hdfc.bank.in "Regalia Gold" MITC
site:hdfc.bank.in "Regalia Gold" Fees
site:hdfc.bank.in "Regalia Gold" Lounge
site:hdfc.bank.in "Regalia Gold" Rewards
```

Output:

```json
{
  "product_page": "...",
  "mitc_pdf": "...",
  "fees_pdf": "...",
  "rewards_pdf": "...",
  "lounge_pdf": "..."
}
```

---

## 2. Document Collector Agent

Downloads:

```text
HTML
PDF
T&C
FAQ
```

Stores:

```text
raw_documents/
```

---

## 3. Document Parser Agent

Converts:

```text
PDF → Text
HTML → Text
FAQ → Text
```

Output:

```json
{
  "source": "MITC",
  "text": "...."
}
```

---

## 4. Feature Extraction Agent

LLM Prompt:

```text
Extract:

joining_fee
annual_fee
forex_markup
reward_rate
lounge_access
fuel_benefits
travel_benefits
movie_benefits

Return JSON only.
```

Output:

```json
{
  "annual_fee": 2500,
  "forex_markup": 2,
  "lounge_access": 12
}
```

---

## 5. Validation Agent

Checks:

```text
MITC vs Product Page
Fees Page vs Product Page
Rewards Page vs T&C
```

Confidence:

```json
{
  "annual_fee": {
      "value":2500,
      "confidence":0.98
  }
}
```

---

# Tier 1: User Understanding

```text
User Profiler Agent
        ↓
Spending Analyzer Agent
        ↓
Financial Twin Agent
```

---

## User Profiler Agent

Input:

```text
Age
Income
Occupation
City
Travel Frequency
```

Output:

```json
{
  "income": 1200000,
  "travel_profile": "high"
}
```

---

## Spending Analyzer Agent

Reads:

```text
Bank Statements
SMS
UPI History
Manual Inputs
```

Categorizes:

```json
{
 "dining": 15000,
 "fuel": 5000,
 "travel": 25000,
 "shopping": 12000
}
```

---

# Tier 2: Card Evaluation Engine

Every card enters the pipeline.

---

## Reward Simulation Agent

Computes:

```text
Expected Reward Value
```

Example:

```text
Dining
Fuel
Travel
Shopping
```

↓

```json
{
  "annual_rewards": 18400
}
```

---

## Lounge Valuation Agent

```text
Visits × Value Per Visit
```

```json
{
  "lounge_value": 7200
}
```

---

## Membership Valuation Agent

Calculates:

```text
Swiggy One
MMT Black
OTT
Golf
```

```json
{
  "membership_value": 4200
}
```

---

## Cost Agent

Calculates:

```text
Annual Fee
Forex Charges
Interest Risk
Hidden Costs
```

```json
{
  "cost": 2500
}
```

---

# Tier 3: Financial Twin Simulation

This is your differentiator.

For each card:

```text
Current Behaviour
      ↓
12-Month Simulation
      ↓
Best Case
Average Case
Worst Case
```

Output:

```json
{
  "best": 22000,
  "avg": 17000,
  "worst": 9500
}
```

---

# Tier 4: Deliberation Layer

Agents debate.

### Cashback Expert

```text
Recommend Millennia
```

### Travel Expert

```text
Recommend Regalia Gold
```

### Premium Expert

```text
Recommend Infinia
```

### Cost Optimizer

```text
Recommend MoneyBack+
```

### Risk Agent

```text
User unlikely to justify annual fee
```

---

# Tier 5: Ranking Engine

Weighted score:

```math
FinalScore =
0.35 × NetAnnualValue
+
0.20 × UserMatch
+
0.15 × ApprovalProbability
+
0.15 × FutureValue
+
0.15 × AgentConsensus
```

FinalScore=0.35(NetAnnualValue)+0.20(UserMatch)+0.15(ApprovalProbability)+0.15(FutureValue)+0.15(AgentConsensus)

---

# Tier 6: Explanation Agent

Output:

```text
1. Regalia Gold earns ₹18,400/year from your spending.

2. You will use 9 of 12 lounge visits.

3. MMT Black membership adds ₹2,000 value.

4. Net expected benefit:
₹15,900 after annual fee.

5. Better than Millennia by ₹4,300.
```

---

# Final Architecture

```text
                        DAILY CRON

Google Discovery
        ↓
Document Collection
        ↓
PDF/HTML Parsing
        ↓
LLM Feature Extraction
        ↓
Validation
        ↓
Card Knowledge Base
        ↓

================================================

User Profile
        ↓
Spending Analyzer
        ↓
Financial Twin
        ↓

Reward Simulator
Lounge Simulator
Membership Simulator
Cost Simulator
        ↓

Multi-Agent Deliberation
        ↓

Ranking Engine
        ↓

Top 5 Cards
        ↓

Explanation Agent
        ↓

User Decision
```

This architecture scales from **100 cards to 1000+ cards** with very little change because the difficult part (document discovery and feature extraction) is separated from the recommendation logic.

