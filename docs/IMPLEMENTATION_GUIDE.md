# CashFlowAdvisor Implementation Guide

This document explains how to structure the project, how to convert notebook logic into Python modules and callable functions, and how to organize memory for a multi-agent financial advisor.

## 1) Target Stack

- **Frontend:** React
- **Server 1:** Node.js for UI-facing API orchestration, auth/session handling, BFF routes, and realtime concerns
- **Server 2:** FastAPI for ML/decision logic, forecasting, simulation, and council reasoning
- **ML/Logic Layer:** Python modules imported by FastAPI

For this project, React should only handle the UI. Node should sit between the UI and the ML service when you want a clean frontend backend. FastAPI should own the AI/ML execution path. Python should hold the business logic that currently lives inside notebooks.

## 2) Recommended Folder Structure

```text
project/
  frontend/
    src/
      components/
      pages/
      hooks/
      api/
      state/

  node-server/
    src/
      routes/
      controllers/
      middleware/
      services/
      utils/
      server.js

  fastapi-server/
    app/
      api/
      councils/
      decision/
      memory/
      schemas/
      services/
      utils/
      main.py
    tests/

  ml/
    notebooks/
      CashFlowAdvisor.ipynb
      cashflow_simulator.ipynb
      financial_forecast_holt_winters.ipynb
    src/
      councils/
      decision/
      memory/
      simulations/
      forecasting/
      cashflow/
      common/
      data/
    scripts/
      train.py
      evaluate.py
      simulate.py

  shared/
    schemas/
    constants/
    prompts/

  infra/
    docker/
    compose.yml
```

## 3) Council Structure

Your council model is a good fit for a modular backend. Each council should be a set of focused Python modules, not one giant agent file.

### Risk Council

- Emergency Fund Agent
- Insurance Agent
- Debt Trap Agent
- Fraud Protection Agent

### Growth Council

- Asset Allocation Agent
- Credit Card Agent
- Loan Advisor Agent
- Retirement Planning Agent

### Benefits Council

- Scheme Matching Agent
- Eligibility Agent

### Behavioral Council

- Bias Detection Agent
- Habit Formation Agent
- Nudge Strategy Agent
- Financial Literacy Agent

### Cashflow Council

- Cashflow Stability Agent
- Income Projection Agent
- Expense Optimizer Agent
- Goal Allocation Agent

### Decision Layer

- Intent Router
- Multi-Agent Deliberation Engine
- Counterfactual Simulator
- Monte Carlo Engine
- Utility Optimizer
- Master Judge

### Memory Layer

- Episodic Memory
- Semantic Memory
- Behavioral Memory
- Goal Memory
- Simulation Memory
- Community Memory

## 4) Notebook-to-Python Conversion Pattern

The main rule is: every stable notebook step becomes a pure function, a class method, or a thin script entrypoint.

### What stays in notebooks

- experiments
- charts
- temporary debugging
- comparing model variants
- inspecting sample outputs

### What moves into `.py`

- data loading
- cleaning
- feature engineering
- forecasting
- simulation
- recommendation scoring
- memory reads/writes
- API-facing orchestration

### Conversion workflow

1. Identify the notebook pipeline stages.
2. Extract repeated code into functions.
3. Move reusable logic into `ml/src/`.
4. Keep I/O separate from logic.
5. Add script entrypoints for repeatable runs.
6. Import those modules from FastAPI and notebooks.

## 5) Notebook Mapping to Python Modules

### `CashFlowAdvisor.ipynb`

Likely contains the main advisor workflow. Split it into:

```text
ml/src/decision/router.py
ml/src/decision/deliberation.py
ml/src/cashflow/forecast.py
ml/src/cashflow/optimization.py
ml/src/memory/store.py
backend/app/api/chat.py
```

### `cashflow_simulator.ipynb`

Move simulation logic to:

```text
ml/src/simulations/cashflow_monte_carlo.py
ml/src/simulations/counterfactuals.py
ml/src/simulations/scenarios.py
```

### `financial_forecast_holt_winters.ipynb`

Move forecasting logic to:

```text
ml/src/forecasting/holt_winters.py
ml/src/forecasting/features.py
ml/src/forecasting/metrics.py
```

## 6) Example Function Breakdown

The notebook should stop owning logic. The logic should live in small Python functions like these:

```python
def load_transactions(path):
    """Load user transaction history from CSV, JSON, or database."""


def clean_transactions(df):
    """Standardize dates, categories, amounts, and missing values."""


def build_cashflow_features(df):
    """Create monthly inflow, outflow, savings, and volatility features."""


def forecast_cashflow(features, horizon_months):
    """Predict future inflows and outflows."""


def run_monte_carlo_simulation(profile, forecast, n_sims):
    """Estimate uncertainty across future financial outcomes."""


def route_intent(user_query, context):
    """Select the relevant council or councils."""


def generate_recommendation(agent_outputs, simulation_results, memory):
    """Return the final response to the user."""
```

### Good conversion rule

- one function = one responsibility
- one module = one topic
- one script = one runnable job

## 7) Call Flow

```text
User Input
  -> Intent Router
  -> Relevant Council(s)
  -> Agent Deliberation
  -> Monte Carlo + Counterfactual Analysis
  -> Master Judge
  -> Personalized Recommendation
  -> Memory Update
```

### Suggested runtime flow

```text
React UI
  -> Node /api/chat
  -> auth/session/rate-limit checks
  -> FastAPI /chat
  -> route_intent()
  -> council agent functions
  -> simulate outcomes
  -> score strategies
  -> master_judge()
  -> response JSON
  -> memory.write()
```

## 8) Server Responsibilities

### Node.js server

Node should handle the UI-facing backend layer.

Use it for:

- authentication and session management
- request validation and rate limiting
- BFF endpoints for React
- websocket or streaming responses if needed
- aggregation of responses from FastAPI and other external APIs

### FastAPI server

FastAPI should expose the AI/ML service layer.

### Example FastAPI endpoints

- `POST /chat` for user questions
- `POST /simulate/cashflow` for scenario testing
- `POST /recommend` for decision support
- `GET /memory/{user_id}` for retrieval
- `POST /memory/update` for persistence

### FastAPI should not do

- notebook-style experimentation
- plotting
- ad hoc model tuning
- business logic duplicated inside route handlers

The route should call functions from `ml/src/` or `fastapi-server/app/services/`.

### Example Node endpoints

- `POST /api/chat` for frontend requests
- `POST /api/login` for auth/session setup
- `GET /api/memory/:user_id` for UI-safe access
- `POST /api/recommendation` to proxy requests to FastAPI

Node should forward ML-heavy requests to FastAPI instead of duplicating model logic.

## 9) Memory Structure

Memory should be explicit and separated by type. Do not mix long-term facts, conversation state, and simulation results in one blob.

### Recommended memory layout

```text
fastapi-server/app/memory/
  episodic.py
  semantic.py
  behavioral.py
  goal.py
  simulation.py
  community.py
  store.py
```

### Memory types

#### Episodic Memory
Stores past conversations, recommendations, and decisions.

Example:
- user asked about emergency fund
- recommended corpus = 6 months
- user rejected high-risk allocation

#### Semantic Memory
Stores stable facts about the user.

Example:
- monthly income
- dependents
- risk tolerance
- age band
- job type

#### Behavioral Memory
Stores spending habits and decision patterns.

Example:
- recurring overspending on weekends
- delayed bill payments
- low savings consistency

#### Goal Memory
Stores long-term objectives.

Example:
- build emergency fund
- buy house in 3 years
- retire by 55

#### Simulation Memory
Stores prior Monte Carlo outputs and scenario results.

Example:
- cashflow stress test results
- probability of goal failure
- best-case / base-case / worst-case paths

#### Community Memory
Stores anonymized cohort patterns.

Example:
- users with similar profiles reduced debt faster with a certain payoff plan

## 10) How Memory Should Be Accessed

The agent flow should read memory before generating advice and write memory after the final judge returns a recommendation.

### Read path

```text
user query
  -> intent router
  -> fetch semantic memory
  -> fetch episodic memory
  -> fetch behavioral and goal memory
  -> council reasoning
```

### Write path

```text
final recommendation
  -> store episode
  -> update user facts if needed
  -> update behavior signals
  -> store simulation output
  -> update goal progress
```

## 11) Suggested Python Module Responsibilities

```text
ml/src/
  councils/
    risk/
    growth/
    benefits/
    behavioral/
    cashflow/
  decision/
    router.py
    deliberation.py
    judge.py
    utility.py
  forecasting/
    holt_winters.py
    features.py
  simulations/
    monte_carlo.py
    counterfactuals.py
  memory/
    store.py
    schemas.py
  common/
    types.py
    config.py
    logging.py
  data/
    loaders.py
    validators.py
```

## 12) Practical Migration Order

1. Move notebook helper code into `ml/src/common/` and `ml/src/data/`.
2. Extract forecast and simulation logic from the notebooks.
3. Wrap those functions in FastAPI endpoints.
4. Add memory read/write modules.
5. Build the council router and deliberation flow.
6. Add React UI that only calls API endpoints.

## 13) Rule of Thumb

If code is reusable, testable, or needed by the API, it belongs in `.py` files.
If code is exploratory, visual, or temporary, it stays in the notebook.

## 14) Minimal Example of the Final Shape

```python
# backend/app/api/chat.py
from backend.app.decision.router import route_intent
from backend.app.decision.judge import master_judge
from backend.app.memory.store import load_user_memory, save_episode


def chat_handler(user_id, message):
    memory = load_user_memory(user_id)
    councils = route_intent(message, memory)
    recommendations = []

    for council in councils:
        recommendations.extend(council.run(message, memory))

    final_answer = master_judge(recommendations, memory)
    save_episode(user_id, message, final_answer)
    return final_answer
```

This gives you a clean separation between UI, orchestration, ML logic, and memory.
