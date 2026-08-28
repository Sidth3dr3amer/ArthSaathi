# Graph Report - Namura - Copy  (2026-08-29)

## Corpus Check
- 263 files · ~358,345 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1430 nodes · 2969 edges · 78 communities (75 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 77 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- CashFlowAdvisor Implementation Guide
- CreditCardAgentWorking.md
- process_input
- test_fraud.py
- chat
- test_profile_agent.py
- test_query_funnel.py
- test_debt_trap.py
- test_eligibility.py
- test_stability.py
- financial-twin
- CLAUDE.md
- state.py
- question_gen.py
- test_retirement.py
- test_recorder.py
- Credit-Card Data Pipeline
- creditcarddatamaker-final
- bias_detection.py
- Financial Twin
- test_router.py
- get_workflow
- test_insurance.py
- test_goal_allocation.py
- test_store.py
- test_expense_optimizer.py
- embeddings.py
- UserProfile
- get_store
- store.py
- InMemoryStore
- Goal
- config.py
- profile.py
- test_loan_advisor.py
- test_asset_allocation.py
- test_credit_card.py
- MemoryStore
- test_features.py
- extractor.py
- test_workflows.py
- features.py
- literacy_advisor
- new_state
- FinancialState
- test_behavioral_agents.py
- habit_formation_advisor
- test_counterfactual_and_utility.py
- load_transactions
- run
- create_memories
- set_store
- test_contracts.py
- orchestrator.py

## God Nodes (most connected - your core abstractions)
1. `new_state()` - 125 edges
2. `UserProfile` - 113 edges
3. `FinancialState` - 60 edges
4. `MemoryStore` - 30 edges
5. `InMemoryStore` - 27 edges
6. `Debt` - 23 edges
7. `route()` - 22 edges
8. `scheme_matching_advisor()` - 21 edges
9. `goal_allocation_advisor()` - 21 edges
10. `emergency_fund_node()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `loaded_state()` --indirect_call--> `goal_allocation_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/cashflow/goal_allocation.py
- `loaded_state()` --indirect_call--> `retirement_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/growth/retirement.py
- `loaded_state()` --indirect_call--> `debt_trap_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/risk/debt_trap.py
- `loaded_state()` --indirect_call--> `emergency_fund_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/risk/emergency_fund.py
- `loaded_state()` --indirect_call--> `insurance_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/risk/insurance.py

## Import Cycles
- None detected.

## Communities (78 total, 3 thin omitted)

### Community 0 - "CashFlowAdvisor Implementation Guide"
Cohesion: 0.04
Nodes (45): 10) How Memory Should Be Accessed, 11) Suggested Python Module Responsibilities, 12) Practical Migration Order, 13) Rule of Thumb, 14) Minimal Example of the Final Shape, 1) Target Stack, 2) Recommended Folder Structure, 3) Council Structure (+37 more)

### Community 1 - "CreditCardAgentWorking.md"
Cohesion: 0.08
Nodes (24): 1. Google Discovery Agent, 2. Document Collector Agent, 3. Document Parser Agent, 4. Feature Extraction Agent, 5. Validation Agent, Cashback Expert, Cost Agent, Cost Optimizer (+16 more)

### Community 2 - "process_input"
Cohesion: 0.10
Nodes (27): extract_information(), Provider, Extract profile fields from processed input. `processed` is the Input…, detect_language(), extract_amounts(), normalise_text(), process_input(), Any (+19 more)

### Community 3 - "test_fraud.py"
Cohesion: 0.06
Nodes (55): analyze_domain(), calculate_risk_score(), check_rbi_alerts(), check_sebi(), detect_mlm_phrases(), detect_scam_phrases(), fraud_node(), gather_evidence() (+47 more)

### Community 4 - "chat"
Cohesion: 0.39
Nodes (7): post, ask_llm(), chat(), detect_voice(), text_to_speech(), transcribe(), UploadFile

### Community 5 - "test_profile_agent.py"
Cohesion: 0.13
Nodes (24): answer_with_context(), compose_draft(), generate_response(), Any, Provider, The complete slide-8 pipeline: InputProcessor -> InformationExtractor ->…, Answer a question using retrieved profile and memory -- the RAG read path, as…, Build the deterministic reply. This is what the user gets if the LLM is down. (+16 more)

### Community 6 - "test_query_funnel.py"
Cohesion: 0.08
Nodes (41): apply_record_filter(), ask(), compute_aggregate(), discover_schema(), explain(), filter_by_domain(), plan_query(), project_fields() (+33 more)

### Community 7 - "test_debt_trap.py"
Cohesion: 0.10
Nodes (36): allocate_payments(), calculate_available_budget(), calculate_available_budget_emergency(), calculate_emi_arrears(), calculate_loss_if_not_paid(), calculate_mandatory_payments(), debt_trap_node(), get_debt_bucket() (+28 more)

### Community 8 - "test_eligibility.py"
Cohesion: 0.08
Nodes (42): _check_rule(), check_scheme(), eligibility_advisor(), eligibility_node(), load_schemes(), Any, Path, Benefits Council -> Eligibility Agent. New agent. Evaluates a user against the… (+34 more)

### Community 9 - "test_stability.py"
Cohesion: 0.07
Nodes (48): cashflow_simulator(), Any, Cashflow Council -> Cashflow Stability Agent. Migrated verbatim from…, Project the user's balance forward and score the resulting risk. Uses the…, Simulate month-by-month cashflow under three scenarios. Returns DataFrame with…, Score and flag risks from the cashflow simulation. Returns a dict of risk flags…, risk_engine(), stability_node() (+40 more)

### Community 12 - "state.py"
Cohesion: 0.09
Nodes (34): detect_recurring(), holt_winters_forecast(), income_projection_node(), Any, Series, Cashflow Council -> Income Projection Agent. Migrated verbatim from…, Project income forward. With enough history, blends Holt-Winters and SARIMAX…, Classify a time-series as recurring if coefficient of variation < threshold.… (+26 more)

### Community 13 - "question_gen.py"
Cohesion: 0.16
Nodes (17): completeness(), generate_question(), _is_unset(), missing_fields(), Any, question_plan(), Profile Agent -> Question Generator. Decides what to ask next. This is the…, Fields in the bank that this profile has not answered, most valuable first. (+9 more)

### Community 14 - "test_retirement.py"
Cohesion: 0.09
Nodes (41): _future_value(), _fv_of_sip(), Any, Growth Council -> Retirement Planning Agent. New agent. Sizes the corpus a user…, LangGraph adapter. Uses the goal allocation agent's leftover surplus as the…, Future value of a monthly contribution stream., Monthly contribution required to reach a target., Size the retirement corpus and the gap. Pure and deterministic. (+33 more)

### Community 15 - "test_recorder.py"
Cohesion: 0.10
Nodes (30): memory_recall_node(), memory_write_node(), _money(), Any, The "Update Memory" step. Every workflow diagram in the deck terminates in a…, One embeddable sentence describing an agent's finding., Persist everything this run produced. The terminal node of every workflow., Load relevant prior context. Runs at the FRONT of a workflow so agents can… (+22 more)

### Community 17 - "Credit-Card Data Pipeline"
Cohesion: 0.40
Nodes (4): Credit-Card Data Pipeline, Directories, Flow, Notes

### Community 20 - "bias_detection.py"
Cohesion: 0.14
Nodes (22): bias_detection_advisor(), bias_detection_node(), detect_impulse_buying(), detect_lifestyle_inflation(), detect_present_bias(), detect_present_focus_on_saving(), detect_subscription_creep(), _finding() (+14 more)

### Community 21 - "Financial Twin"
Cohesion: 0.33
Nodes (5): Financial Twin, Known limitations, Layout, Running, Setup

### Community 29 - "test_router.py"
Cohesion: 0.07
Nodes (48): anthropic_client(), available_providers(), cerebras_client(), chat(), groq_client(), is_configured(), llm7_client(), LLMNotConfigured (+40 more)

### Community 30 - "get_workflow"
Cohesion: 0.13
Nodes (18): get_workflow(), Build (and cache) a workflow graph by name., parametrize, Every flow diagram in the deck terminates in 'Update Memory'., `total_tokens` carries an `operator.add` reducer, so returning the sub-graph's…, Fanning these out concurrently would silently weaken every answer, because the…, test_a_workflow_produces_its_declared_agents_results(), test_declared_stages_appear_in_the_graph() (+10 more)

### Community 46 - "test_insurance.py"
Cohesion: 0.10
Nodes (34): _age_loading(), insurance_advisor(), insurance_node(), Any, Risk Council -> Insurance Agent. New agent (not a migration). Sizes a user's…, Size the protection gap. Pure and deterministic: no I/O, no LLM. Returns per-…, _term_multiple(), advise() (+26 more)

### Community 47 - "test_goal_allocation.py"
Cohesion: 0.11
Nodes (30): goal_allocation_advisor(), goal_allocation_node(), Any, Cashflow Council -> Goal Allocation Agent. New agent (not a migration). Splits…, LangGraph adapter. Consumes the Emergency Fund agent's output when it ran…, Allocate surplus across goals. Pure and deterministic: no I/O, no LLM. `goals`…, Cashflow Council -> Goal Allocation Agent., A huge suggested contribution cannot starve every goal. (+22 more)

### Community 48 - "test_store.py"
Cohesion: 0.08
Nodes (9): _pg_store(), populated(), fixture, parametrize, Memory Layer -> MemoryStore contract. Every test here runs TWICE: once against…, Yields each implementation in turn, isolated to a unique user id., store(), test_all_six_memory_types_are_writable() (+1 more)

### Community 49 - "test_expense_optimizer.py"
Cohesion: 0.12
Nodes (27): expense_optimizer_advisor(), expense_optimizer_node(), Any, Cashflow Council -> Expense Optimizer Agent. New agent (not a migration).…, Find recoverable overspend. Pure and deterministic: no I/O, no LLM., advise(), parametrize, Cashflow Council -> Expense Optimizer Agent. (+19 more)

### Community 50 - "embeddings.py"
Cohesion: 0.21
Nodes (12): cosine_similarity(), embed(), embed_many(), hashed_embedding(), _normalise(), openai_embedding(), Embeddings for semantic memory recall. Pluggable by design, because none of the…, Embed one string using the configured backend. (+4 more)

### Community 51 - "UserProfile"
Cohesion: 0.09
Nodes (27): computed_field, build_context(), profile_summary(), Any, Assemble the prompt context, trimming the weakest memories to fit the budget., Retriever and Context Builder in one call., A compact, readable profile block. Only states what is actually known., retrieve_and_build() (+19 more)

### Community 52 - "get_store"
Cohesion: 0.17
Nodes (15): get_store(), The process-wide store. Uses Postgres when DATABASE_URL is set, otherwise falls…, load_profile(), Any, Profile Agent -> Profile Updater. Merges extracted fields into the stored…, Persist the profile document. Never raises into a conversation., Load a stored profile, or None when the user is new or the store is down., Merge an extraction into a profile and persist the result. (+7 more)

### Community 53 - "store.py"
Cohesion: 0.12
Nodes (12): ABC, DeclarativeBase, Base, Memory, Any, SQLAlchemy models for the memory layer (Neon Postgres + pgvector). The deck…, Durable profile document maintained by the Profile Agent., UserProfileRecord (+4 more)

### Community 54 - "InMemoryStore"
Cohesion: 0.16
Nodes (6): InMemoryStore, Dict-backed store with identical semantics to the Postgres one., fixture, A state carrying one real agent result., run_state(), store()

### Community 55 - "Goal"
Cohesion: 0.16
Nodes (15): BaseModel, Goal, A financial goal the user is saving toward., history_profile(), indebted_profile(), fixture, Shared pytest fixtures. Design rule: no test in this suite may require a…, All-zero boundary case. Nothing may divide by zero or return NaN. (+7 more)

### Community 56 - "config.py"
Cohesion: 0.20
Nodes (9): env(), Central configuration and path resolution. Every module resolves paths through…, Read an environment variable, treating blank strings as unset., Read a required environment variable or fail with an actionable message., require_env(), active_backend(), Neon hands out a libpq URL; SQLAlchemy needs an explicit driver.…, _sqlalchemy_url() (+1 more)

### Community 57 - "profile.py"
Cohesion: 0.09
Nodes (38): _annual_value(), _need_weight(), Any, Benefits Council -> Scheme Matching Agent. New agent. Eligibility answers "can…, Rank schemes this user should actually pursue. Pure and deterministic., Indicative annual rupee value the user actually realises. Combines the payout…, How much this user's situation calls for this scheme. Returns (weight, reasons)., scheme_matching_advisor() (+30 more)

### Community 58 - "test_loan_advisor.py"
Cohesion: 0.10
Nodes (36): emi(), loan_advisor_advisor(), loan_advisor_node(), max_principal(), Any, Growth Council -> Loan Advisor Agent. New agent. Answers three separate…, Standard reducing-balance EMI., Invert the EMI formula to get the borrowable principal. (+28 more)

### Community 59 - "test_asset_allocation.py"
Cohesion: 0.07
Nodes (46): asset_allocation_advisor(), asset_allocation_node(), Any, Growth Council -> Asset Allocation Agent. New agent. Produces a target…, LangGraph adapter. Consumes the Emergency Fund and Debt Trap agents when they…, Recommend a target allocation. Pure and deterministic: no I/O, no LLM., emergency_fund_advisor(), emergency_fund_node() (+38 more)

### Community 60 - "test_credit_card.py"
Cohesion: 0.08
Nodes (48): analyze_spend_profile(), build_llm_context(), build_spend_routing(), calculate_card_value(), credit_card_node(), filter_eligible_cards(), load_card_database(), profile_to_engine_dict() (+40 more)

### Community 61 - "MemoryStore"
Cohesion: 0.10
Nodes (15): MemoryStore, Any, What agents may do with memory., Store one memory and return it., Semantic search over a user's memories, most similar first., Most recently created memories, newest first., Delete a user's memories. Returns how many were removed., Profile Agent -> Memory Creator. Writes what was learned in a conversation turn… (+7 more)

### Community 62 - "test_features.py"
Cohesion: 0.09
Nodes (8): feats(), fixture, Behavioral Council -> shared feature extraction. Asserted against the seeded…, One unusual final month must not masquerade as a trend., The spike was planted in dining and entertainment only. A blended average…, test_per_category_timing_recovers_the_planted_spike(), test_trend_compares_thirds_not_endpoints(), txns()

### Community 63 - "extractor.py"
Cohesion: 0.16
Nodes (14): _coerce(), parse_extraction(), Any, Profile Agent -> Information Extractor. Pulls structured profile fields out of…, Would this value survive UserProfile validation?, Coerce a raw model value to the declared type, or reject it., Parse and filter a model response. Tolerates the model wrapping JSON in prose…, _validates() (+6 more)

### Community 64 - "test_workflows.py"
Cohesion: 0.15
Nodes (9): order_agents(), Sort a set of agents into dependency order, dropping unknown names., Workflows -> the eight flows from the deck, plus the orchestrator. The…, Derived from AGENT_ORDER rather than the union of the other workflows -- that…, Deliberation is expensive; a card question has a computable answer., test_cheap_questions_do_not_convene_the_councils(), test_full_review_runs_every_agent(), test_ordering_a_subset_preserves_dependency_order() (+1 more)

### Community 65 - "features.py"
Cohesion: 0.17
Nodes (28): _avg(), category_totals(), credits(), _day(), _days_in_month(), debits(), extract_features(), impulse_clusters() (+20 more)

### Community 66 - "literacy_advisor"
Cohesion: 0.16
Nodes (15): _lesson(), literacy_advisor(), literacy_node(), Any, Behavioral Council -> Financial Literacy Agent. Works out which concepts this…, Identify literacy gaps from behaviour. Pure and deterministic., LangGraph adapter. Consumes Bias Detection, Emergency Fund and Benefits…, test_a_well_managed_profile_has_few_gaps() (+7 more)

### Community 67 - "new_state"
Cohesion: 0.08
Nodes (48): build_deliberation_graph(), count_tokens(), deliberate(), _findings_for(), make_advisor(), make_critic(), make_judge(), Persona (+40 more)

### Community 68 - "FinancialState"
Cohesion: 0.18
Nodes (14): _advice_of(), FinancialState, agent_node(), build_workflow(), make_agent_runner(), Any, Workflow construction. Every workflow in the deck has the same spine: Memory…, Assemble a workflow graph. Returns a compiled LangGraph whose terminal node is… (+6 more)

### Community 69 - "test_behavioral_agents.py"
Cohesion: 0.10
Nodes (23): _nudge(), nudge_strategy_advisor(), nudge_strategy_node(), Any, Behavioral Council -> Nudge Strategy Agent. Turns detected patterns into…, LangGraph adapter. Consumes Bias Detection and Habit Formation upstream., Build a ranked, capped nudge programme. Pure and deterministic., bias() (+15 more)

### Community 70 - "habit_formation_advisor"
Cohesion: 0.16
Nodes (15): _habit(), habit_formation_advisor(), habit_formation_node(), observed_routines(), Any, Behavioral Council -> Habit Formation Agent. Identifies what the user already…, LangGraph adapter. Consumes Bias Detection's findings when available., Regularities already present in the user's behaviour. (+7 more)

### Community 71 - "test_counterfactual_and_utility.py"
Cohesion: 0.07
Nodes (53): _apply(), counterfactual_advisor(), counterfactual_node(), default_scenarios(), _diff(), _evaluate(), Any, Decision Layer -> Counterfactual Simulator. Answers "what if I did X instead?"… (+45 more)

### Community 72 - "load_transactions"
Cohesion: 0.19
Nodes (13): date, generate_transactions(), load_transactions(), Any, Path, Synthetic transaction generator for the Behavioral Council. The repo has no…, Load the seeded dataset, returning [] when it is absent., Generate and persist the dataset. (+5 more)

### Community 73 - "run"
Cohesion: 0.18
Nodes (12): Any, Provider, Route a query to the right workflow and run it end to end. Returns the final…, A compact, JSON-safe view of a run for an API response or a dashboard.…, run(), summarise_run(), test_a_full_review_activates_every_agent(), test_a_narrow_query_activates_few_agents() (+4 more)

### Community 74 - "create_memories"
Cohesion: 0.25
Nodes (9): create_memories(), phrase(), Any, Turn a field/value into an embeddable sentence fragment., Write semantic memories for learned facts and one episodic memory for the turn.…, test_learned_facts_become_semantic_memories(), test_memory_creation_survives_a_broken_store(), test_phrasing_falls_back_for_unknown_fields() (+1 more)

### Community 75 - "set_store"
Cohesion: 0.33
Nodes (6): Override the process-wide store (used by tests and the API layer)., set_store(), offline(), profile(), fixture, No workflow test may hit a network.

### Community 76 - "test_contracts.py"
Cohesion: 0.15
Nodes (15): Debt, One liability. Field names match `DebtOvercomeAdvisor` exactly — notably `emi`…, parametrize, Cross-cutting contracts every agent must honour. These are the guard-rails that…, No numpy scalars, no pandas objects, no datetimes may leak out of an agent. The…, No agent may raise on a brand-new user with no data., test_new_state_seeds_the_collection_fields(), test_node_does_not_mutate_the_profile() (+7 more)

### Community 77 - "orchestrator.py"
Cohesion: 0.40
Nodes (4): The eight workflows from the deck's orchestration diagram. Each is a…, The workflow a routed intent should run., workflow_for_intent(), The Decision Orchestrator. Top-level entry point: a user message and a profile…

## Knowledge Gaps
- **65 isolated node(s):** `creditcarddatamaker-final`, `financial-twin`, `graphify`, `Flow`, `Directories` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserProfile` connect `UserProfile` to `test_profile_agent.py`, `test_debt_trap.py`, `test_eligibility.py`, `state.py`, `question_gen.py`, `test_retirement.py`, `test_insurance.py`, `test_goal_allocation.py`, `test_expense_optimizer.py`, `get_store`, `Goal`, `profile.py`, `test_loan_advisor.py`, `test_asset_allocation.py`, `test_credit_card.py`, `MemoryStore`, `extractor.py`, `test_workflows.py`, `literacy_advisor`, `new_state`, `FinancialState`, `test_behavioral_agents.py`, `test_counterfactual_and_utility.py`, `run`, `set_store`, `test_contracts.py`, `orchestrator.py`?**
  _High betweenness centrality (0.222) - this node is a cross-community bridge._
- **Why does `new_state()` connect `new_state` to `test_fraud.py`, `test_debt_trap.py`, `test_eligibility.py`, `test_stability.py`, `state.py`, `test_retirement.py`, `test_recorder.py`, `bias_detection.py`, `test_router.py`, `get_workflow`, `test_insurance.py`, `test_goal_allocation.py`, `test_expense_optimizer.py`, `UserProfile`, `InMemoryStore`, `profile.py`, `test_loan_advisor.py`, `test_asset_allocation.py`, `test_credit_card.py`, `test_workflows.py`, `literacy_advisor`, `FinancialState`, `test_behavioral_agents.py`, `habit_formation_advisor`, `test_counterfactual_and_utility.py`, `run`, `test_contracts.py`, `orchestrator.py`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `FinancialState` connect `FinancialState` to `test_fraud.py`, `test_debt_trap.py`, `test_eligibility.py`, `test_stability.py`, `state.py`, `test_retirement.py`, `test_recorder.py`, `bias_detection.py`, `test_router.py`, `test_insurance.py`, `test_goal_allocation.py`, `test_expense_optimizer.py`, `UserProfile`, `profile.py`, `test_loan_advisor.py`, `test_asset_allocation.py`, `test_credit_card.py`, `literacy_advisor`, `new_state`, `test_behavioral_agents.py`, `habit_formation_advisor`, `test_counterfactual_and_utility.py`, `orchestrator.py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `UserProfile` (e.g. with `FinancialState` and `new_state()`) actually correct?**
  _`UserProfile` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `MemoryStore` (e.g. with `memory_recall_node()` and `memory_write_node()`) actually correct?**
  _`MemoryStore` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `InMemoryStore` (e.g. with `store()` and `test_a_failing_store_does_not_break_the_workflow()`) actually correct?**
  _`InMemoryStore` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `creditcarddatamaker-final`, `financial-twin`, `graphify` to the rest of the system?**
  _65 weakly-connected nodes found - possible documentation gaps or missing edges._