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
   → Intent Router                 which councils does this actually need?
   → Memory Recall                 what do we already know about this person?
   → Agents (dependency order)     the councils that were routed to
   → Monte Carlo / Counterfactual  what if they did something else?
   → Council Deliberation          5 councils argue, a judge synthesises
   → Utility Optimizer             split the surplus by marginal utility
   → Memory Update                 remember it for next time
```

A narrow question is cheap — *"which credit card should I get?"* activates **one**
agent and returns in ~0.5s. *"Give me a full financial review"* activates all
**18** and convenes every council.

---

## Status

**18 council agents · 9 workflows · 6 memory types · 25 government schemes ·
52 modules · 17 endpoints · 1,049 tests (1,023 offline, 26 live)**

| Layer | What's in it | State |
|---|---|---|
| **Risk Council** | Emergency Fund · Insurance · Debt Trap · Fraud Protection | complete |
| **Growth Council** | Asset Allocation · Credit Card · Loan Advisor · Retirement | complete |
| **Cashflow Council** | Stability · Income Projection · Expense Optimizer · Goal Allocation | complete |
| **Benefits Council** | Eligibility · Scheme Matching | complete |
| **Behavioral Council** | Bias Detection · Habit Formation · Nudge Strategy · Financial Literacy | complete |
| **Decision Layer** | Intent Router · Deliberation · Master Judge · Monte Carlo · Counterfactual · Utility Optimizer · Query Funnel | complete |
| **Memory Layer** | Episodic · Semantic · Behavioural · Goal · Simulation · Community — Neon Postgres + pgvector | complete, lexical embeddings |
| **Profile Agent** | Input Processor · Extractor · Updater · Memory Creator · Question Gen · Response Gen · RAG | complete |
| **Credit-Card Tiers** | Tier 1 Profiling → 2 Evaluation → 3 Twin Simulation → 4 Expert Panel → 5 Ranking → 6 Explanation | complete |
| **Workflows** | 9 end-to-end flows, each terminating in a memory write | complete |
| **API** | FastAPI, 17 endpoints over the whole system | complete |
| **Frontend** | React dashboard — Report, Schemes, Onboarding, Voice | complete |

---

## How each agent works

Every agent is a pure function of the profile. No I/O, no LLM, no hidden state —
the logic below *is* the agent, and the LLM only ever narrates the result.

### Risk Council

| Agent | Logic |
|---|---|
| **Emergency Fund** | Buffer-stock sizing. Months of runway needed is scaled by income volatility, dependants and job type, then tiered into cash / liquid / short-term. Returns the target, the gap, and the monthly contribution that closes it. |
| **Insurance** | Sizes three protection gaps. Term life by Human Life Value (10–15× annual income scaled by age — younger earners need a larger multiple — plus outstanding debt, minus existing cover), health as a floater sized by dependants and age band, critical illness at roughly a year of income past 35. Ranked by **shortfall ratio × criticality**, not rupee magnitude, or term cover in crores would always outrank health in lakhs. |
| **Debt Trap** | Ranks debts by effective rate (nominal + penalties + fees), then allocates every spare rupee to the top-ranked debt while holding minimums everywhere else. Migrated byte-identical from the verified notebook. |
| **Fraud** | Three separated layers: network evidence (domain age, RBI/SEBI registries, complaints, news), pure phrase detection (scam and MLM patterns), and scoring. Network calls fire **only** when the query actually looks like a fraud question — a guard that took a routine review from 61s to 0.00s. |

### Growth Council

| Agent | Logic |
|---|---|
| **Asset Allocation** | Two-factor. Risk **capacity** (horizon, income stability, dependants, runway, debt load — computed, not asked) versus risk **tolerance** (stated preference). Capacity binds, so someone with no emergency fund does not get an equity-heavy split because they said they were aggressive. |
| **Credit Card** | Spend profile → hard eligibility gate (age, income, employment) → net annual value (rewards earned − fees) → per-category spend routing. **Checks the cards the user already holds first:** someone revolving a balance is told to clear it rather than sold a second card. |
| **Loan Advisor** | Separates three questions borrowers conflate: what a lender *will* give (FOIR headroom), what they *should* borrow (affordability), and whether to prepay or invest (rate arbitrage). Banks underwrite to roughly 50–55% FOIR, and the gap between the first two is where households get into trouble. |
| **Retirement** | Inflates today's essential expenses to the retirement date, sizes the corpus on the **real** (inflation-adjusted) return during drawdown, and solves for the monthly contribution. Avoids the flat "25× expenses" rule, which silently assumes a 4% real withdrawal and breaks under inflation. |

### Cashflow Council

| Agent | Logic |
|---|---|
| **Income Projection** | Detects which streams are stable enough to treat as recurring (coefficient of variation below a threshold), then forecasts with Holt-Winters triple-exponential smoothing and SARIMAX, and blends the two. |
| **Stability** | Month-by-month balance projection under base / optimistic / pessimistic scenarios, scored 0–100 with human-readable flags. |
| **Expense Optimizer** | Benchmarks each category against a healthy share of income and ranks by what is *realistically recoverable* — rent cannot be cut this month, dining can — so the headline saving is achievable rather than theoretical. |
| **Goal Allocation** | Splits surplus across goals, and when they do not fit reports **two** honest options rather than one: keep every target and push the deadlines out, or keep the deadlines and cut the targets. |

### Benefits Council

| Agent | Logic |
|---|---|
| **Eligibility** | A rule engine over 25 machine-readable central schemes, returning an auditable trace. **Three verdicts, not two** — a rule it cannot evaluate returns `unknown`, never a guess, plus the missing field that would unlock the most schemes. A rule engine rather than an LLM because eligibility is a legal determination. |
| **Scheme Matching** | `benefit × need × effort`. Benefit is the annual rupee value, log-scaled so a ₹1 crore credit line does not drown out a ₹6,000 transfer the user will actually claim; need is user-specific; effort is penalised per required document, because paperwork is why entitlements go unclaimed. |

### Behavioral Council

All four reason over the same derived features, computed once from transaction history.

| Agent | Logic |
|---|---|
| **Bias Detection** | Finds patterns and **prices them**. Every finding carries the evidence that produced it and an annual rupee cost — *"your dining spend runs 1.8× higher in the last five days of the month, which costs about ₹14,000 a year"*, not *"you have present bias"*. |
| **Habit Formation** | Finds regularities that already exist (a salary credit on the 1st, a weekly grocery run) and anchors each suggestion to one as an implementation intention. New habits stick to existing cues, not to willpower. |
| **Nudge Strategy** | Ranks and **caps** interventions, because attention is the scarce resource and sending five is worse than sending one. A nudge changes the choice architecture — default, timing, friction, framing — rather than issuing advice. |
| **Financial Literacy** | Infers gaps from behaviour, not from a quiz. Someone paying 42% revolving interest while holding idle cash has a specific, identifiable gap, and the lesson shows them the arithmetic on their own balance. |

---

## How deliberation works

The agents produce numbers. Deliberation is where the councils **argue about what
those numbers mean**, because the right answer is usually contested: the Risk
Council wants the surplus in an emergency fund, Growth wants it compounding, and
each is right in isolation.

It is a 9-node LangGraph — five advisors, five critics, one judge.

```
          ┌── Risk ───────┐         ┌── critiques Growth ────┐
          ├── Growth ─────┤         ├── critiques Cashflow ──┤
findings ─┼── Cashflow ───┼────────▶┼── critiques Behavioral ┼──▶ Judge ──▶ plan
          ├── Behavioral ─┤         ├── critiques Benefits ──┤
          └── Benefits ───┘         └── critiques Risk ──────┘
             argue (60w)               cross-critique (40w)    synthesise (150w)
```

**1. Each council argues its own case.** A persona is *data*, not code — a `key`,
a `brief`, the council it speaks for, and `result_keys` naming which agent
outputs ground it. That last field is what keeps the debate honest: a council
argues from its own agents' computed findings, so the Risk Council cites the
actual runway figure rather than improvising one. Briefs are capped at
**60 words, plain prose, no markdown**.

**2. Round-robin cross-criticism.** Each council critiques a *different* council —
never itself, and never symmetrically:

```
risk → growth → cashflow → behavioral → benefits → risk
```

The pairing is deliberate rather than arbitrary. Risk versus Growth is the
central tension (protect now versus compound later), and Behavioral sits where it
can ask whether any of it will actually be acted on. Critiques are capped at
**40 words**.

**3. A judge synthesises.** It sees every brief and every critique and produces
one prioritised recommendation, capped at **150 words**.

**Design decisions worth naming:**

- **The councils fan out in parallel**, under `operator.add` reducers on
  `verdicts`, `critiques`, `errors` and `total_tokens`. This is the one place in
  the system where parallelism is correct, because the five arguments are
  genuinely independent of each other.
- **Errors propagate.** A council that fails is not silently dropped: the result
  says *"only 3 of 5 councils convened"* rather than presenting a partial debate
  as a complete one.
- **Every prompt is bounded.** Unbounded council and judge output once produced a
  16,434px page. Word caps plus UI clamping brought it to 2,120px.
- **Token accounting returns deltas**, not accumulated totals — the sub-graph is
  seeded with parent state, so returning the running total double-counted it.
- **Provider failover.** `DELIBERATION_PROVIDER` selects the LLM; on a missing key
  *or* a live failure the call reroutes once via `PROVIDER_FALLBACK` and reports
  the original exception. Bounded concurrency and rate-limit retry sit under it.

The same graph runs with three generic personas (conservative / growth / value)
as a fallback when a bare query arrives with no council results in state.

---

## Some things worth knowing

**Every agent core is a pure function.** `<agent>_advisor(...) -> dict` does no
I/O and calls no LLM; a thin `<agent>_node(state)` adapts it to LangGraph, and
LLM narration is a separate function. This is why the suite runs offline, and why
the Counterfactual Simulator can re-run the *same* code against a modified
profile — a "what if" can never drift from the real answer.

**Agents run in dependency order, not in parallel.** Goal allocation reserves
runway using the emergency fund's output; nudges target the biases actually
detected. Fanning them out would be faster and would silently produce weaker
answers, since each agent would see an empty state. `AGENT_ORDER` is the single
declaration of that order, and seven of the dependencies are asserted in tests.

**Eligibility returns three verdicts, not two.** Telling someone they qualify for
PM-KISAN and having the bank reject them is a real harm.

**Benefits are classified by kind before anything is summed.** Income,
protection, credit access and savings capacity are four different things, and
only income reaches the headline. Adding them together once told a street vendor
earning ₹2.16 lakh that she qualified for ₹7.26 lakh a year — 335% of her income
— by counting a ₹1 crore *loan* and a ₹25 lakh subsidy *ceiling* as income.

**Contingent benefits are not valued at their headline number.** PMSBY is a
₹20/year accident policy with ₹2,00,000 of cover; valuing the cover directly
ranked it above PM-KISAN's *guaranteed* ₹6,000 transfer.

**A card is not recommended to someone revolving a balance.** Carrying ₹48,000 at
42% costs about ₹20,160 a year — roughly 11× the best card's net value. The agent
says so and declines, rather than presenting a reward gain that the interest bill
dwarfs. Cards the user already holds are also filtered out of the ranking.

**Suspicious profile writes are held, not applied.** A misheard income poisons
every downstream council, so a change of more than 3× on a critical field is
surfaced as a question instead of being written.

---

## The card database

31 cards back the recommendation engine: **4 hand-checked** in
`final_decision/`, plus **27 promoted** from the 148 LLM-extracted profiles in
`card_attributes/` by `ml/src/cards/promote.py`.

The other 121 are deliberately skipped. A card is promoted only when its profile
states a reward rate on *ordinary* spend; category and merchant rates are not
enough. Three earlier versions of the parser proved why — they promoted Axis
Magnus at a 15% base earn rate (a milestone bonus), HDFC Pixel at 5% (a marketing
line contradicting its own *"1% CashBack on all transactions"*), and twelve more
cards at 5% on all spend, read out of lines like *"10% cashback on Samsung
purchases"*. Every one of those outranked all four hand-checked cards. A wrong
rate does not fail loudly; it silently mis-ranks every recommendation it touches.

Promoted cards carry `eligibility_confirmed: false` and stay permissive on
employment rules — the raw `eligibility` field contains card *networks*, which is
extraction noise — so the flag travels through the API rather than the gap being
guessed at. A hand-checked card always wins a name collision, and the golden
tests are pinned against the curated four, so promotion can add candidates but
can never move an existing card's score.

---

## Running it

```bash
uv sync                     # or: pip install -e .
cp .env.example .env        # then fill in your keys
```

**Required:** `GROQ_API_KEY` (most agents), `DATABASE_URL` (Neon Postgres with
`CREATE EXTENSION vector`). **Optional:** `OPENROUTER_API_KEY`, `LLM7_API_KEY`
(deliberation), `CEREBRAS_API_KEY` (card extraction), `ANTHROPIC_API_KEY` (falls
back to Groq if unset), `TAVILY_API_KEY` / `SERPAPI_KEY` (fraud search falls back
to DuckDuckGo).

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

**The dashboard**

```bash
cd frontend && npm install && npm run dev    # http://localhost:5173
```

Four screens: the **Report** (ask anything, watch the councils argue), **Schemes**,
**Onboarding** (the Teaching Saathis conversation), and **Voice**. It proxies
`/api` to the backend on :8000, so no CORS setup is needed.

**The voice assistant** (separate service, multilingual — Hindi, Marathi,
Kannada, English):

```bash
cd TestVoice && uvicorn backend:app --port 8000   # then open index.html
```

**Tests**

```bash
pytest -m "not live"     # 1,023, fully offline
pytest -m live           # 26, requires DATABASE_URL and network
```

---

## Layout

```
ml/src/
  common/        config, LLM providers, synthetic transaction generator
  schemas/       UserProfile and the unified FinancialState
  councils/      the 18 agents, by council
  decision/      router, deliberation, counterfactual, monte carlo, utility
  memory/        MemoryStore (Postgres + in-memory), embeddings, recorder
  profile_agent/ the 6-stage onboarding pipeline + RAG
  cards/         credit-card Tiers 1-6 + promote.py
  workflows/     the 9 flows + orchestrator
ml/data/         25 government schemes, synthetic transactions
server/          FastAPI app, 17 endpoints
frontend/        React dashboard (Vite)
tests/           36 test files, one per agent
docs/            architecture and credit-card agent design
```

Notebooks (`CashFlowAdvisor/`, `FraudDetectionAdvisor/`, …) are **preserved demo
evidence**, not the running system. Each carries a banner naming the module that
superseded it — editing a notebook will not change behaviour.

---

## Known limitations

- **Embeddings are lexical, not semantic.** No configured provider offers an
  embeddings API and PyTorch is not installed, so the default backend is a
  deterministic hashed bag-of-words. It ranks correctly, but *"emergency fund"*
  will not match *"rainy day savings"*. Set `EMBEDDING_BACKEND=openai` with an
  `OPENAI_API_KEY` for true semantic recall.
- **Scheme data is hand-curated** as of 2026-08 and drifts with budget cycles.
  Verify amounts against the official portals before relying on them.
- **The transaction dataset is synthetic**, with behavioural signals deliberately
  planted so the Behavioral Council can be tested against known ground truth.
- **121 of the 148 extracted cards cannot be scored.** Their profiles state no
  reward rate on ordinary spend. This is a limit of the Tier-0 extraction, not of
  the engine — the fix is better extraction, not a more permissive parser.
- **Promoted cards have unverified eligibility.** They are permissive on
  employment and income rules and flagged `eligibility_confirmed: false`, so a
  user can be shown a card they would in fact be rejected for.
- **Council latency is variable.** A full deliberation runs 8–60s depending on
  provider load. Narrow questions return in under a second.
- **Retirement can recommend an unaffordable SIP.** A gig worker already in
  deficit is still told the contribution that closes the corpus gap, with no
  check against current surplus. The number is arithmetically right and
  practically useless.
- **A farmer still gets few card options.** `self_employed` is excluded by most
  curated cards. The promoted pool widens this, but eligibility there is
  unconfirmed.
