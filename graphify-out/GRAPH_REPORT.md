# Graph Report - Namura - Copy  (2026-08-29)

## Corpus Check
- 332 files · ~385,899 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2062 nodes · 4257 edges · 108 communities (95 shown, 13 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 96 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7a4cbce6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- CashFlowAdvisor Implementation Guide
- CreditCardAgentWorking.md
- test_profile_agent.py
- test_fraud.py
- chat
- chat
- test_query_funnel.py
- test_asset_allocation.py
- test_eligibility.py
- test_stability.py
- financial-twin
- CLAUDE.md
- test_income_projection.py
- UserProfile
- test_retirement.py
- new_state
- Credit-Card Data Pipeline
- creditcarddatamaker-final
- bias_detection.py
- ArthaSaathi
- test_router.py
- test_workflows.py
- test_insurance.py
- test_goal_allocation.py
- test_store.py
- test_expense_optimizer.py
- embeddings.py
- rag.py
- get_store
- MemoryStore
- run_state
- Goal
- RuntimeError
- test_scheme_matching.py
- test_loan_advisor.py
- emergency_fund_node
- test_credit_card.py
- test_tier1_profiler.py
- test_features.py
- llm.py
- test_tier6_and_pipeline.py
- features.py
- literacy_advisor
- test_deliberation.py
- FinancialState
- test_behavioral_agents.py
- habit_formation_advisor
- test_counterfactual_and_utility.py
- test_promote.py
- test_tier2_evaluation.py
- test_tier5_ranking.py
- test_tier4_experts.py
- test_contracts.py
- App.jsx
- test_api.py
- cards/conftest.py
- test_tier3_twin.py
- package.json
- statistical_estimator
- chat.py
- credit_card_node
- existing_card_burden
- memory.py
- stability_node
- load_card_database
- counterfactual.py
- chat
- warmup
- parametrize
- test_forecasters_fall_back_to_the_mean_on_unfittable_input
- client
- trending
- loaded_state
- _diminishing
- server/__init__.py
- test_an_inline_profile_cannot_write_into_another_users_record
- test_ask_answers_from_memory
- test_questions_work_for_a_brand_new_user
- test_a_revolving_user_is_warned_rather_than_sold_a_second_card
- test_a_user_with_no_card_balance_still_gets_recommendations
- test_voice_status_does_not_load_the_model
- test_health_never_leaks_key_values

## God Nodes (most connected - your core abstractions)
1. `new_state()` - 141 edges
2. `UserProfile` - 139 edges
3. `FinancialState` - 64 edges
4. `MemoryStore` - 34 edges
5. `InMemoryStore` - 32 edges
6. `get_store()` - 32 edges
7. `run_card_intelligence()` - 30 edges
8. `credit_card_node()` - 29 edges
9. `scheme_matching_advisor()` - 26 edges
10. `profile_to_engine_dict()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `test_memory_degrades_on_store_outage()` --uses--> `InMemoryStore`  [INFERRED]
  tests/server/test_api.py → ml/src/memory/store.py
- `test_evaluating_no_cards_returns_nothing()` --calls--> `evaluate_all()`  [EXTRACTED]
  tests/cards/test_tier2_evaluation.py → ml/src/cards/tier2_evaluation.py
- `test_simulate_all_covers_every_card()` --calls--> `simulate_all()`  [EXTRACTED]
  tests/cards/test_tier3_twin.py → ml/src/cards/tier3_twin.py
- `loaded_state()` --indirect_call--> `goal_allocation_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/cashflow/goal_allocation.py
- `loaded_state()` --indirect_call--> `retirement_node()`  [INFERRED]
  tests/decision/test_counterfactual_and_utility.py → ml/src/councils/growth/retirement.py

## Import Cycles
- None detected.

## Communities (108 total, 13 thin omitted)

### Community 0 - "CashFlowAdvisor Implementation Guide"
Cohesion: 0.04
Nodes (45): 10) How Memory Should Be Accessed, 11) Suggested Python Module Responsibilities, 12) Practical Migration Order, 13) Rule of Thumb, 14) Minimal Example of the Final Shape, 1) Target Stack, 2) Recommended Folder Structure, 3) Council Structure (+37 more)

### Community 1 - "CreditCardAgentWorking.md"
Cohesion: 0.08
Nodes (24): 1. Google Discovery Agent, 2. Document Collector Agent, 3. Document Parser Agent, 4. Feature Extraction Agent, 5. Validation Agent, Cashback Expert, Cost Agent, Cost Optimizer (+16 more)

### Community 2 - "test_profile_agent.py"
Cohesion: 0.05
Nodes (76): _coerce(), extract_information(), parse_extraction(), Any, Provider, Would this value survive UserProfile validation?, Extract profile fields from processed input. `processed` is the Input…, Coerce a raw model value to the declared type, or reject it. (+68 more)

### Community 3 - "test_fraud.py"
Cohesion: 0.06
Nodes (55): analyze_domain(), calculate_risk_score(), check_rbi_alerts(), check_sebi(), detect_mlm_phrases(), detect_scam_phrases(), fraud_node(), gather_evidence() (+47 more)

### Community 4 - "chat"
Cohesion: 0.39
Nodes (7): ask_llm(), chat(), detect_voice(), post, text_to_speech(), transcribe(), UploadFile

### Community 5 - "chat"
Cohesion: 0.06
Nodes (49): Exception, anthropic_client(), _attempt(), _call(), cerebras_client(), chat(), is_configured(), _is_rate_limit() (+41 more)

### Community 6 - "test_query_funnel.py"
Cohesion: 0.06
Nodes (49): live, groq_client(), apply_record_filter(), ask(), compute_aggregate(), discover_schema(), explain(), filter_by_domain() (+41 more)

### Community 7 - "test_asset_allocation.py"
Cohesion: 0.06
Nodes (63): asset_allocation_advisor(), asset_allocation_node(), Any, LangGraph adapter. Consumes the Emergency Fund and Debt Trap agents when they…, Recommend a target allocation. Pure and deterministic: no I/O, no LLM., allocate_payments(), calculate_available_budget(), calculate_available_budget_emergency() (+55 more)

### Community 8 - "test_eligibility.py"
Cohesion: 0.07
Nodes (43): _check_rule(), check_scheme(), eligibility_advisor(), eligibility_node(), load_schemes(), Any, Path, Evaluate one scheme. Every rule is reported, not just the failing one. (+35 more)

### Community 9 - "test_stability.py"
Cohesion: 0.20
Nodes (18): Score and flag risks from the cashflow simulation. Returns a dict of risk flags…, risk_engine(), Cashflow Council -> Cashflow Stability Agent., `income_values` was a notebook global; it must default, not raise., `today` is a parameter, so labels are deterministic in tests., _risk(), _sim(), test_balance_compounds_month_over_month() (+10 more)

### Community 12 - "test_income_projection.py"
Cohesion: 0.15
Nodes (23): detect_recurring(), holt_winters_forecast(), income_projection_node(), Any, Series, Cashflow Council -> Income Projection Agent. Migrated verbatim from…, Project income forward. With enough history, blends Holt-Winters and SARIMAX…, Classify a time-series as recurring if coefficient of variation < threshold.… (+15 more)

### Community 13 - "UserProfile"
Cohesion: 0.09
Nodes (36): completeness(), generate_question(), _is_unset(), missing_fields(), Any, question_plan(), Profile Agent -> Question Generator. Decides what to ask next. This is the…, Fields in the bank that this profile has not answered, most valuable first. (+28 more)

### Community 14 - "test_retirement.py"
Cohesion: 0.09
Nodes (41): _future_value(), _fv_of_sip(), Any, Growth Council -> Retirement Planning Agent. New agent. Sizes the corpus a user…, LangGraph adapter. Uses the goal allocation agent's leftover surplus as the…, Future value of a monthly contribution stream., Monthly contribution required to reach a target., Size the retirement corpus and the gap. Pure and deterministic. (+33 more)

### Community 15 - "new_state"
Cohesion: 0.10
Nodes (33): memory_recall_node(), memory_write_node(), _money(), Any, One embeddable sentence describing an agent's finding., Persist everything this run produced. The terminal node of every workflow., Load relevant prior context. Runs at the FRONT of a workflow so agents can…, summarise() (+25 more)

### Community 17 - "Credit-Card Data Pipeline"
Cohesion: 0.40
Nodes (4): Credit-Card Data Pipeline, Directories, Flow, Notes

### Community 20 - "bias_detection.py"
Cohesion: 0.14
Nodes (22): bias_detection_advisor(), bias_detection_node(), detect_impulse_buying(), detect_lifestyle_inflation(), detect_present_bias(), detect_present_focus_on_saving(), detect_subscription_creep(), _finding() (+14 more)

### Community 21 - "ArthaSaathi"
Cohesion: 0.25
Nodes (7): ArthaSaathi, Known limitations, Layout, Running it, Some things worth knowing, The system, What it does

### Community 29 - "test_router.py"
Cohesion: 0.10
Nodes (31): agents_for(), classify_by_llm(), classify_by_rules(), Any, Provider, Deterministic classification. Returns (intent, matched pattern text)., LLM fallback. Never raises -- a failure or a hallucinated label degrades to…, Agents to activate. An empty agent list means every agent in the councils. (+23 more)

### Community 30 - "test_workflows.py"
Cohesion: 0.06
Nodes (51): agent_node(), make_agent_runner(), order_agents(), Any, Import an agent's node function by name., Sort a set of agents into dependency order, dropping unknown names., Build one node that runs the given agents in dependency order. A failing agent…, get_workflow() (+43 more)

### Community 46 - "test_insurance.py"
Cohesion: 0.10
Nodes (34): _age_loading(), insurance_advisor(), insurance_node(), Any, Risk Council -> Insurance Agent. New agent (not a migration). Sizes a user's…, Size the protection gap. Pure and deterministic: no I/O, no LLM. Returns per-…, _term_multiple(), advise() (+26 more)

### Community 47 - "test_goal_allocation.py"
Cohesion: 0.11
Nodes (30): goal_allocation_advisor(), goal_allocation_node(), Any, Cashflow Council -> Goal Allocation Agent. New agent (not a migration). Splits…, LangGraph adapter. Consumes the Emergency Fund agent's output when it ran…, Allocate surplus across goals. Pure and deterministic: no I/O, no LLM. `goals`…, Cashflow Council -> Goal Allocation Agent., A huge suggested contribution cannot starve every goal. (+22 more)

### Community 48 - "test_store.py"
Cohesion: 0.05
Nodes (17): InMemoryStore, Dict-backed store with identical semantics to the Postgres one., _pg_store(), populated(), fixture, parametrize, Memory Layer -> MemoryStore contract. Every test here runs TWICE: once against…, Yields each implementation in turn, isolated to a unique user id. (+9 more)

### Community 49 - "test_expense_optimizer.py"
Cohesion: 0.12
Nodes (27): expense_optimizer_advisor(), expense_optimizer_node(), Any, Cashflow Council -> Expense Optimizer Agent. New agent (not a migration).…, Find recoverable overspend. Pure and deterministic: no I/O, no LLM., advise(), parametrize, Cashflow Council -> Expense Optimizer Agent. (+19 more)

### Community 50 - "embeddings.py"
Cohesion: 0.17
Nodes (15): env(), Read an environment variable, treating blank strings as unset., active_backend(), cosine_similarity(), embed(), embed_many(), hashed_embedding(), _normalise() (+7 more)

### Community 51 - "rag.py"
Cohesion: 0.13
Nodes (18): build_context(), profile_summary(), Any, Profile Agent -> Retriever + Context Builder. The RAG half of the deck's…, Assemble the prompt context, trimming the weakest memories to fit the budget., Retriever and Context Builder in one call., Fetch the profile and the memories most relevant to the query. Never raises: a…, A compact, readable profile block. Only states what is actually known. (+10 more)

### Community 52 - "get_store"
Cohesion: 0.06
Nodes (58): FastAPI, Central configuration and path resolution. Every module resolves paths through…, get_store(), The process-wide store. Uses Postgres when DATABASE_URL is set, otherwise falls…, load_profile(), Profile Agent -> Profile Updater. Merges extracted fields into the stored…, Load a stored profile, or None when the user is new or the store is down., Workflow construction. Every workflow in the deck has the same spine: Memory… (+50 more)

### Community 53 - "MemoryStore"
Cohesion: 0.07
Nodes (22): ABC, DeclarativeBase, Base, Memory, Any, SQLAlchemy models for the memory layer (Neon Postgres + pgvector). The deck…, Durable profile document maintained by the Profile Agent., UserProfileRecord (+14 more)

### Community 54 - "run_state"
Cohesion: 0.50
Nodes (4): fixture, A state carrying one real agent result., run_state(), store()

### Community 55 - "Goal"
Cohesion: 0.11
Nodes (18): computed_field, Goal, BaseModel, Income minus essential expenses and mandatory debt payments., Monthly debt servicing as a fraction of monthly income., A financial goal the user is saving toward., history_profile(), indebted_profile() (+10 more)

### Community 56 - "RuntimeError"
Cohesion: 0.22
Nodes (8): Read a required environment variable or fail with an actionable message., require_env(), Neon hands out a libpq URL; SQLAlchemy needs an explicit driver.…, _sqlalchemy_url(), RuntimeError, test_ask_degrades_without_a_provider(), test_profile_turn_degrades_on_pipeline_failure(), test_workflow_degrades_on_failure()

### Community 57 - "test_scheme_matching.py"
Cohesion: 0.07
Nodes (48): _annual_value(), _need_weight(), Any, Indicative annual rupee value the user actually realises. Combines payout…, How much this user's situation calls for this scheme. Returns (weight, reasons)., Rank schemes this user should actually pursue. Pure and deterministic., scheme_matching_advisor(), scheme_matching_node() (+40 more)

### Community 58 - "test_loan_advisor.py"
Cohesion: 0.10
Nodes (35): emi(), loan_advisor_advisor(), loan_advisor_node(), max_principal(), Any, Standard reducing-balance EMI., Invert the EMI formula to get the borrowable principal., Assess borrowing capacity and the prepay-versus-invest trade-off. Pure and… (+27 more)

### Community 59 - "emergency_fund_node"
Cohesion: 0.18
Nodes (18): emergency_fund_advisor(), emergency_fund_node(), Any, Risk Council -> Emergency Fund Agent. Migrated verbatim from…, LangGraph adapter. Reads the unified profile, writes one result key., Size a user's emergency fund and the monthly contribution needed to reach it.…, parametrize, Risk Council -> Emergency Fund Agent. The golden-value test below pins the… (+10 more)

### Community 60 - "test_credit_card.py"
Cohesion: 0.17
Nodes (20): calculate_card_value(), filter_eligible_cards(), profile_to_engine_dict(), Adapt the unified `UserProfile` to the flat dict the engine expects. Field…, Returns (eligible_cards, rejected_cards_with_reason)., Estimate annual monetary value a user gets from a card. Returns a breakdown…, parametrize, Growth Council -> Credit Card Agent (Tier 2: Card Evaluation Engine). Golden… (+12 more)

### Community 61 - "test_tier1_profiler.py"
Cohesion: 0.09
Nodes (48): financial_twin_advisor(), _income_band(), Any, Map raw spend into card-reward buckets and annualise. Pure and deterministic.…, The behavioural model Tier 2 values benefits against. Pure and deterministic.…, Run the three Tier 1 agents in order., Adapter. Writes one result key., Build the cardholder profile. Pure and deterministic. Mirrors the doc's… (+40 more)

### Community 62 - "test_features.py"
Cohesion: 0.07
Nodes (21): date, generate_transactions(), load_transactions(), Any, Path, Synthetic transaction generator for the Behavioral Council. The repo has no…, Load the seeded dataset, returning [] when it is absent., Generate and persist the dataset. (+13 more)

### Community 63 - "llm.py"
Cohesion: 0.10
Nodes (21): _cashback_score(), _cost_optimizer_score(), Expert, _premium_score(), Any, Credit Card Intelligence -> Tier 4: Deliberation Layer. Cashback Expert |…, Confidence that this card is SAFE for this user. Low means the Risk Agent…, One voice in the Tier 4 debate. (+13 more)

### Community 64 - "test_tier6_and_pipeline.py"
Cohesion: 0.06
Nodes (42): card_intelligence_node(), Any, Provider, Adapter. Writes one result key., Run all six tiers. Deterministic by default (`use_llm=False`), so the ranking…, run_card_intelligence(), build_explanation(), explain() (+34 more)

### Community 65 - "features.py"
Cohesion: 0.17
Nodes (28): _avg(), category_totals(), credits(), _day(), _days_in_month(), debits(), extract_features(), impulse_clusters() (+20 more)

### Community 66 - "literacy_advisor"
Cohesion: 0.16
Nodes (15): _lesson(), literacy_advisor(), literacy_node(), Any, Behavioral Council -> Financial Literacy Agent. Works out which concepts this…, Identify literacy gaps from behaviour. Pure and deterministic., LangGraph adapter. Consumes Bias Detection, Emergency Fund and Benefits…, test_a_well_managed_profile_has_few_gaps() (+7 more)

### Community 67 - "test_deliberation.py"
Cohesion: 0.08
Nodes (47): _advice_of(), build_deliberation_graph(), count_tokens(), deliberate(), _findings_for(), make_advisor(), make_critic(), make_judge() (+39 more)

### Community 68 - "FinancialState"
Cohesion: 0.13
Nodes (16): Credit Card Intelligence -> the full Tier 1-6 pipeline. User Profile ->…, Credit Card Intelligence -> Tier 1: User Understanding. User Profiler Agent ->…, Benefits Council -> Eligibility Agent. New agent. Evaluates a user against the…, Benefits Council -> Scheme Matching Agent. New agent. Eligibility answers "can…, Growth Council -> Asset Allocation Agent. New agent. Produces a target…, build_llm_context(), Growth Council -> Credit Card Agent (also Tier 2: Card Evaluation Engine).…, Serialize the algorithmic outputs into a compact JSON payload for the LLM. (+8 more)

### Community 69 - "test_behavioral_agents.py"
Cohesion: 0.10
Nodes (23): _nudge(), nudge_strategy_advisor(), nudge_strategy_node(), Any, Behavioral Council -> Nudge Strategy Agent. Turns detected patterns into…, LangGraph adapter. Consumes Bias Detection and Habit Formation upstream., Build a ranked, capped nudge programme. Pure and deterministic., bias() (+15 more)

### Community 70 - "habit_formation_advisor"
Cohesion: 0.16
Nodes (15): _habit(), habit_formation_advisor(), habit_formation_node(), observed_routines(), Any, Behavioral Council -> Habit Formation Agent. Identifies what the user already…, LangGraph adapter. Consumes Bias Detection's findings when available., Regularities already present in the user's behaviour. (+7 more)

### Community 71 - "test_counterfactual_and_utility.py"
Cohesion: 0.10
Nodes (36): counterfactual_advisor(), Run each scenario against the same agents and diff against the baseline.…, build_claims(), Any, Allocate surplus across claims. Pure and deterministic., LangGraph adapter. Consumes whatever councils ran upstream., Turn council outputs into competing claims on the surplus. Only agents that…, utility_advisor() (+28 more)

### Community 72 - "test_promote.py"
Cohesion: 0.08
Nodes (43): _base_rate(), _category_rates(), _lounge_visits(), _number(), promote_all(), promote_card(), Any, Path (+35 more)

### Community 73 - "test_tier2_evaluation.py"
Cohesion: 0.07
Nodes (38): cost_agent_advisor(), evaluate_card(), lounge_valuation_advisor(), membership_valuation_advisor(), Any, Value of bundled memberships and lifestyle perks, discounted by likely use.…, Total annual cost: fee, forex, interest risk, hidden costs. Pure and…, Run all four Tier 2 agents against one card. (+30 more)

### Community 74 - "test_tier5_ranking.py"
Cohesion: 0.08
Nodes (27): approval_probability_score(), future_value_score(), _normalise(), Any, rank_cards(), Credit Card Intelligence -> Tier 5: Ranking Engine. FinalScore = 0.35 x…, Durability of the card's value. Built on Tier 3's WORST case, not its average,…, Apply the doc's weighted formula. Pure and deterministic. Returns cards ranked… (+19 more)

### Community 75 - "test_tier4_experts.py"
Cohesion: 0.08
Nodes (29): deliberate(), explain_expert(), Provider, Score every card from one expert's viewpoint and name their pick., Phrase one expert's argument. Falls back to a deterministic sentence when no…, Run the full panel. Pure unless `with_arguments=True`, which adds LLM prose.…, run_expert(), analyze_spend_profile() (+21 more)

### Community 76 - "test_contracts.py"
Cohesion: 0.24
Nodes (7): Debt, One liability. Field names match `DebtOvercomeAdvisor` exactly — notably `emi`…, Cross-cutting contracts every agent must honour. These are the guard-rails that…, test_new_state_seeds_the_collection_fields(), test_profile_handles_zero_income_without_dividing_by_zero(), test_profile_round_trips_through_json(), test_profile_surplus_subtracts_mandatory_debt_servicing()

### Community 77 - "App.jsx"
Cohesion: 0.11
Nodes (21): App(), Identity(), TABS, Chamber(), inline(), Prose(), AGENT_COUNCIL, api (+13 more)

### Community 79 - "cards/conftest.py"
Cohesion: 0.15
Nodes (19): evaluate_all(), Credit Card Intelligence -> Tier 2: Card Evaluation Engine. Reward Simulation |…, Evaluate every card, best net value first., Credit Card Intelligence -> Tier 3: Financial Twin Simulation. Current…, Simulate every evaluated card., simulate_all(), card_db(), card_user() (+11 more)

### Community 80 - "test_tier3_twin.py"
Cohesion: 0.15
Nodes (20): Any, Project one card's net value across three scenarios. Pure and deterministic.…, simulate_card(), Credit Card Tier 3 -> 12-month Financial Twin simulation., Spending 20% more clears the threshold; 20% less misses it., test_a_card_with_no_waiver_always_charges_the_fee(), test_a_negative_downside_is_flagged(), test_a_shorter_horizon_scales_the_result_down() (+12 more)

### Community 81 - "package.json"
Cohesion: 0.11
Nodes (18): dependencies, react, react-dom, devDependencies, vite, @vitejs/plugin-react, name, private (+10 more)

### Community 82 - "statistical_estimator"
Cohesion: 0.16
Nodes (17): Series, Decision Layer -> Monte Carlo Engine. Migrated verbatim from…, Monte Carlo estimator for discretionary / irregular spending. Returns: mean…, statistical_estimator(), fixture, Decision Layer -> Monte Carlo Engine. The engine is seeded, so every assertion…, Spending is clipped at zero even when sigma dwarfs the mean., The sample mean should sit closer to the true mean with more draws. (+9 more)

### Community 83 - "chat.py"
Cohesion: 0.18
Nodes (16): _as_line(), ask(), chat(), normalise_recommendations(), Any, post, The main conversational entry point. `POST /chat` is the whole system in one…, Answer from stored profile and memory only. No councils, no simulation. (+8 more)

### Community 84 - "credit_card_node"
Cohesion: 0.12
Nodes (16): build_spend_routing(), credit_card_node(), For each spend category, recommend the best eligible card. Applies travel alpha…, Rank the card database against the user's spend profile. `cards` may be…, Ordering must hold over the full pool, not just the curated four., Pinned against the hand-checked four. Promoting cards into `card_pool/` adds…, Confirms the previous test's empty result is the fee gate, not a bug., test_node_contract_and_counts() (+8 more)

### Community 85 - "existing_card_burden"
Cohesion: 0.15
Nodes (16): existing_card_burden(), Any, What the user's CURRENT cards already cost them. Pure and deterministic. A…, A gold loan at 12% is a debt, but it is not a credit card., The headline failure this guards: recommending a second card to someone paying…, The guard must not suppress advice for someone who carries no balance., _revolver(), test_a_card_the_user_already_holds_is_not_recommended_back() (+8 more)

### Community 86 - "memory.py"
Cohesion: 0.22
Nodes (13): delete, forget(), memory_types(), Any, get, post, Memory endpoints. Read and search a user's memories, and delete them. The…, Most recent memories, newest first. (+5 more)

### Community 87 - "stability_node"
Cohesion: 0.15
Nodes (13): cashflow_simulator(), Any, Cashflow Council -> Cashflow Stability Agent. Migrated verbatim from…, Project the user's balance forward and score the resulting risk. Uses the…, Simulate month-by-month cashflow under three scenarios. Returns DataFrame with…, stability_node(), The two cashflow agents must compose: stability reuses projection's output., test_node_consumes_the_upstream_income_forecast() (+5 more)

### Community 88 - "load_card_database"
Cohesion: 0.17
Nodes (13): load_card_database(), Path, Load the card database: hand-curated cards first, then the promoted pool.…, card_user(), db(), full_db(), fixture, Mirrors the notebook's USER_PROFILE literal. (+5 more)

### Community 89 - "counterfactual.py"
Cohesion: 0.23
Nodes (11): _apply(), counterfactual_node(), default_scenarios(), _diff(), _evaluate(), Any, Decision Layer -> Counterfactual Simulator. Answers "what if I did X instead?"…, Scenarios worth asking about for most users, scaled to their own numbers. (+3 more)

### Community 90 - "chat"
Cohesion: 0.20
Nodes (10): chat(), parametrize, The credit-card agent returns card OBJECTS in its `recommendations`; the route…, An agent-layer failure must not surface as a stack trace., test_chat_degrades_instead_of_500(), test_chat_full_review_runs_every_agent_and_deliberates(), test_chat_recommendations_are_display_strings(), test_chat_routes_to_the_right_workflow() (+2 more)

### Community 91 - "warmup"
Cohesion: 0.25
Nodes (9): load_voice_app(), Any, get, post, Import `TestVoice/backend.py` on first use. Loaded by path rather than as a…, Whether the voice service can be loaded, without loading it., Force the Whisper load now rather than on a user's first request. Worth calling…, voice_status() (+1 more)

### Community 92 - "parametrize"
Cohesion: 0.25
Nodes (8): parametrize, No numpy scalars, no pandas objects, no datetimes may leak out of an agent. The…, No agent may raise on a brand-new user with no data., test_node_does_not_mutate_the_profile(), test_node_only_writes_keys_the_state_declares(), test_node_output_is_json_serialisable(), test_node_survives_an_all_zero_profile(), test_node_writes_its_declared_result_key()

### Community 93 - "test_forecasters_fall_back_to_the_mean_on_unfittable_input"
Cohesion: 0.50
Nodes (4): parametrize, Both wrap their model in try/except and degrade to a flat mean., test_forecasters_fall_back_to_the_mean_on_unfittable_input(), test_forecasters_never_return_negative_income()

### Community 94 - "client"
Cohesion: 0.50
Nodes (4): client(), fixture, A TestClient with no network and no database., store()

### Community 95 - "trending"
Cohesion: 0.67
Nodes (3): fixture, Series, trending()

### Community 96 - "loaded_state"
Cohesion: 0.67
Nodes (3): loaded_state(), profile(), fixture

## Knowledge Gaps
- **80 isolated node(s):** `creditcarddatamaker-final`, `name`, `private`, `version`, `type` (+75 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UserProfile` connect `UserProfile` to `test_profile_agent.py`, `test_asset_allocation.py`, `test_eligibility.py`, `test_income_projection.py`, `test_retirement.py`, `new_state`, `test_workflows.py`, `test_insurance.py`, `test_goal_allocation.py`, `test_store.py`, `test_expense_optimizer.py`, `rag.py`, `get_store`, `Goal`, `test_scheme_matching.py`, `test_loan_advisor.py`, `test_credit_card.py`, `test_tier1_profiler.py`, `llm.py`, `test_tier6_and_pipeline.py`, `literacy_advisor`, `FinancialState`, `test_behavioral_agents.py`, `test_counterfactual_and_utility.py`, `test_contracts.py`, `cards/conftest.py`, `chat.py`, `credit_card_node`, `existing_card_burden`, `load_card_database`, `counterfactual.py`, `loaded_state`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Why does `new_state()` connect `new_state` to `test_fraud.py`, `test_asset_allocation.py`, `test_eligibility.py`, `test_stability.py`, `test_income_projection.py`, `UserProfile`, `test_retirement.py`, `bias_detection.py`, `test_router.py`, `test_workflows.py`, `test_insurance.py`, `test_goal_allocation.py`, `test_expense_optimizer.py`, `get_store`, `run_state`, `test_scheme_matching.py`, `test_loan_advisor.py`, `emergency_fund_node`, `test_credit_card.py`, `test_tier1_profiler.py`, `llm.py`, `test_tier6_and_pipeline.py`, `literacy_advisor`, `test_deliberation.py`, `FinancialState`, `test_behavioral_agents.py`, `habit_formation_advisor`, `test_counterfactual_and_utility.py`, `test_contracts.py`, `credit_card_node`, `existing_card_burden`, `stability_node`, `load_card_database`, `counterfactual.py`, `parametrize`, `loaded_state`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `FinancialState` connect `FinancialState` to `test_fraud.py`, `test_asset_allocation.py`, `test_eligibility.py`, `test_income_projection.py`, `UserProfile`, `test_retirement.py`, `new_state`, `bias_detection.py`, `test_router.py`, `test_workflows.py`, `test_insurance.py`, `test_goal_allocation.py`, `test_expense_optimizer.py`, `get_store`, `MemoryStore`, `test_scheme_matching.py`, `test_loan_advisor.py`, `emergency_fund_node`, `test_tier1_profiler.py`, `llm.py`, `test_tier6_and_pipeline.py`, `literacy_advisor`, `test_deliberation.py`, `test_behavioral_agents.py`, `habit_formation_advisor`, `test_counterfactual_and_utility.py`, `credit_card_node`, `stability_node`, `counterfactual.py`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `UserProfile` (e.g. with `FinancialState` and `new_state()`) actually correct?**
  _`UserProfile` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `MemoryStore` (e.g. with `memory_recall_node()` and `memory_write_node()`) actually correct?**
  _`MemoryStore` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `InMemoryStore` (e.g. with `store()` and `test_a_failing_store_does_not_break_the_workflow()`) actually correct?**
  _`InMemoryStore` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `creditcarddatamaker-final`, `name`, `private` to the rest of the system?**
  _80 weakly-connected nodes found - possible documentation gaps or missing edges._