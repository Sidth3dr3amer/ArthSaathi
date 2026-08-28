# ArthaSaathi

**A multi-agent financial intelligence system for the Indian market.**

Five councils of specialist agents analyse a user's finances, argue with each
other, and produce a single prioritised plan — grounded in a persistent memory of
everything the user has told the system before.

Built by **Team Sailors** — Mayur Raonang and Siddarth Potdar, SPIT Mumbai.

---

## What it does

Ask it a question in plain language. An intent router decides which councils are
needed, runs only those agents, optionally simulates alternative futures, convenes
the councils to deliberate, allocates the user's surplus across competing claims,
and writes what it learned to memory.

```
User query
   → Intent Router            which councils does this actually need?
   → Memory Recall            what do we already know about this person?
   → Agents (dependency order) the councils that were routed to
   → Monte Carlo / Counterfactual   what if they did something else?
   → Council Deliberation     5 councils argue, a judge synthesises
   → Utility Optimizer        split the surplus by marginal utility
   → Memory Update            remember it for next time
```

A narrow question is cheap — *"which credit card should I get?"* activates **one**
agent and returns in ~0.5s. *"Give me a full financial review"* activates all
**18** and convenes every council.

---

## The system

| Layer | What's in it |
|---|---|
| **Risk Council** | Emergency Fund · Insurance · Debt Trap · Fraud Protection |
| **Growth Council** | Asset Allocation · Credit Card · Loan Advisor · Retirement |
| **Cashflow Council** | Stability · Income Projection · Expense Optimizer · Goal Allocation |
| **Benefits Council** | Scheme Matching · Eligibility |
| **Behavioral Council** | Bias Detection · Habit Formation · Nudge Strategy · Financial Literacy |
| **Decision Layer** | Intent Router · Deliberation Engine · Master Judge · Monte Carlo · Counterfactual Simulator · Utility Optimizer |
| **Memory Layer** | Episodic · Semantic · Behavioural · Goal · Simulation · Community — on Neon Postgres + pgvector |
| **Profile Agent** | Input Processor · Information Extractor · Profile Updater · Memory Creator · Question Generator · Response Generator · RAG |
| **Credit-Card Tiers** | Tier 1 Profiling → Tier 2 Evaluation → Tier 3 Twin Simulation → Tier 4 Expert Panel → Tier 5 Ranking → Tier 6 Explanation |
| **Workflows** | 9 end-to-end flows, each terminating in a memory write |
| **API** | FastAPI, 17 endpoints over the whole system |

**18 council agents · 9 workflows · 6 memory types · 25 government schemes · 51 modules · 988 tests**

---

## Some things worth knowing

**Every agent core is a pure function.** `<agent>_advisor(...) -> dict` does no
I/O and calls no LLM; a thin `<agent>_node(state)` adapts it to LangGraph, and
LLM narration is a separate function. This is why the whole suite runs offline in
about 70 seconds, and why the Counterfactual Simulator can re-run the *same* code
against a modified profile — a "what if" can never drift from the real answer.

**Agents run in dependency order, not in parallel.** Goal allocation reserves
runway using the emergency fund's output; nudges target the biases actually
detected. Fanning them out would be faster and would silently produce weaker
answers, since each agent would see an empty state. Seven of these dependencies
are asserted in tests.

**Eligibility returns three verdicts, not two.** A scheme rule it cannot evaluate
returns `unknown`, never a guess — plus the missing field that would unlock the
most schemes. Telling someone they qualify for PM-KISAN and having the bank
reject them is a real harm.

**Contingent benefits are not valued at their headline number.** PMSBY is a
₹20/year accident policy with ₹2,00,000 of cover. Valuing the cover directly
ranked it above PM-KISAN's *guaranteed* ₹6,000 transfer, which is plainly wrong,
so benefits carry a realisation factor by type.

**Suspicious profile writes are held, not applied.** A misheard income poisons
every downstream council, so a change of more than 3× on a critical field is
surfaced as a question instead of being written.

---

## Running it

```bash
uv sync                     # or: pip install -e .
cp .env.example .env        # then fill in your keys
```

**Required:** `GROQ_API_KEY` (most agents), `DATABASE_URL` (Neon Postgres with
`CREATE EXTENSION vector`). **Optional:** `CEREBRAS_API_KEY` (card extraction),
`ANTHROPIC_API_KEY` (falls back to Groq if unset), `LLM7_API_KEY`,
`TAVILY_API_KEY` / `SERPAPI_KEY` (fraud search falls back to DuckDuckGo).

**The API**

```bash
uvicorn server.main:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"demo","message":"which credit card should I get?"}'
```

**In Python**

```python
from ml.src.schemas.profile import UserProfile
from ml.src.workflows.orchestrator import run, summarise_run

profile = UserProfile(user_id="demo", monthly_income=95_000, essential_expenses=48_000)
print(summarise_run(run("give me a full financial review", profile)))
```

**The voice assistant** (separate service, multilingual — Hindi, Marathi,
Kannada, English):

```bash
cd TestVoice && uvicorn backend:app --port 8000   # then open index.html
```

**Tests**

```bash
pytest -m "not live"     # 964, fully offline
pytest -m live           # 24, requires DATABASE_URL
```

---

## Layout

```
ml/src/
  common/       config, LLM providers, synthetic transaction generator
  schemas/      UserProfile and the unified FinancialState
  councils/     the 18 agents, by council
  decision/     router, deliberation, counterfactual, monte carlo, utility
  memory/       MemoryStore (Postgres + in-memory), embeddings, recorder
  profile_agent/ the 6-stage onboarding pipeline + RAG
  cards/        credit-card Tiers 1-6
  workflows/    the 9 flows + orchestrator
ml/data/        25 government schemes, synthetic transactions
server/         FastAPI app
tests/          33 test files, one per agent
docs/           architecture and credit-card agent design
```

Notebooks (`CashFlowAdvisor/`, `FraudDetectionAdvisor/`, …) are **preserved demo
evidence**, not the running system. Each carries a banner naming the module that
superseded it — editing a notebook will not change behaviour.

---

## Known limitations

- **Embeddings are lexical, not semantic.** No configured LLM provider offers an
  embeddings API and PyTorch is not installed, so the default backend is a
  deterministic hashed bag-of-words. It ranks correctly but *"emergency fund"*
  will not match *"rainy day savings"*. Set `EMBEDDING_BACKEND=openai` with an
  `OPENAI_API_KEY` for true semantic recall.
- **Scheme data is hand-curated** as of 2026-08 and drifts with budget cycles.
  Verify amounts against the official portals before relying on them.
- **The transaction dataset is synthetic**, with behavioural signals deliberately
  planted so the Behavioral Council can be tested against known ground truth.
- **Only 4 curated cards** back the recommendation engine, though 148 extracted
  card profiles exist in `CreditCardDataMaker_Final/card_attributes/`.
- **No frontend dashboard yet** — the API and the voice UI exist; the reporting
  screens do not.
