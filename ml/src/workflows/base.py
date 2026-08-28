"""
Workflow construction.

Every workflow in the deck has the same spine:

    Memory Recall -> Agents -> [Simulation] -> [Deliberation] -> Utility -> Memory Update

so it is built once here and parameterised, rather than written out eight times.

**Agents run in dependency order, not in parallel.** Several agents consume an
upstream agent's output -- goal allocation reserves runway using the emergency
fund result, asset allocation reads the debt position, the nudge strategy needs
the detected biases. Fanning them out concurrently would silently produce
weaker answers, because each would see an empty state instead of its input. The
order below is the topological sort of those dependencies; `AGENT_ORDER` is the
single place it is declared.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Sequence

from langgraph.graph import END, START, StateGraph

from ..memory.recorder import memory_recall_node, memory_write_node
from ..schemas.state import COUNCIL_AGENTS, FinancialState

#: Dependency-ordered execution. Earlier agents feed later ones:
#:   emergency_fund -> goal_allocation, asset_allocation, nudge_strategy, literacy
#:   debt_trap      -> asset_allocation, utility
#:   bias_detection -> habit_formation -> nudge_strategy, literacy
#:   income_projection -> stability
#:   goal_allocation   -> retirement
AGENT_ORDER: tuple[str, ...] = (
    "emergency_fund",
    "debt_trap",
    "insurance",
    "fraud",
    "income_projection",
    "stability",
    "expense_optimizer",
    "goal_allocation",
    "asset_allocation",
    "credit_card",
    "loan_advisor",
    "retirement",
    "eligibility",
    "scheme_matching",
    "bias_detection",
    "habit_formation",
    "nudge_strategy",
    "literacy",
)

_AGENT_COUNCIL = {a: c for c, agents in COUNCIL_AGENTS.items() for a in agents}


def agent_node(name: str) -> Callable[[FinancialState], dict[str, Any]]:
    """Import an agent's node function by name."""
    council = _AGENT_COUNCIL.get(name)
    if council is None:
        raise ValueError(f"unknown agent {name!r}")
    module = importlib.import_module(f"..councils.{council}.{name}", __package__)
    return getattr(module, f"{name}_node")


def order_agents(agents: Sequence[str]) -> list[str]:
    """Sort a set of agents into dependency order, dropping unknown names."""
    wanted = set(agents)
    return [a for a in AGENT_ORDER if a in wanted]


def make_agent_runner(agents: Sequence[str]) -> Callable[[FinancialState], dict[str, Any]]:
    """
    Build one node that runs the given agents in dependency order.

    A failing agent is recorded and skipped rather than aborting the workflow --
    a fraud lookup timing out must not cost the user their cashflow analysis.
    """
    ordered = order_agents(agents)
    nodes = [(name, agent_node(name)) for name in ordered]

    def run_agents(state: FinancialState) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        errors: list[str] = []
        working = dict(state)

        for name, node in nodes:
            try:
                result = node(working)
            except Exception as exc:
                errors.append(f"{name}: {exc!r}")
                continue
            patch.update(result)
            working.update(result)      # later agents see earlier results

        if errors:
            patch["errors"] = errors
        return patch

    run_agents.__name__ = "agents"
    return run_agents


def build_workflow(
    name: str,
    agents: Sequence[str],
    *,
    simulate: bool = False,
    deliberate: bool = False,
    optimise: bool = True,
    recall_limit: int = 5,
):
    """
    Assemble a workflow graph.

    Returns a compiled LangGraph whose terminal node is always the memory update,
    matching every flow diagram in the deck.
    """
    from ..decision.counterfactual import counterfactual_node
    from ..decision.utility import utility_node

    builder = StateGraph(FinancialState)

    builder.add_node("recall", lambda s: memory_recall_node(s, limit=recall_limit))
    builder.add_node("agents", make_agent_runner(agents))

    sequence = ["recall", "agents"]

    if simulate:
        builder.add_node("simulate", counterfactual_node)
        sequence.append("simulate")

    if optimise:
        builder.add_node("optimise", utility_node)
        sequence.append("optimise")

    if deliberate:
        from ..decision.deliberation import COUNCIL_PERSONAS, build_deliberation_graph

        personas_used = COUNCIL_PERSONAS
        council_graph = build_deliberation_graph(personas_used)

        def deliberate_node(state: FinancialState) -> dict[str, Any]:
            try:
                out = council_graph.invoke(state)
            except Exception as exc:
                return {"errors": [f"deliberation: {exc!r}"]}

            # `verdicts`, `critiques` and `total_tokens` carry `operator.add`
            # reducers. The sub-graph is seeded with this state, so its output
            # already includes whatever the parent held -- returning that whole
            # value would add it a second time. Return only the delta.
            before_verdicts = len(state.get("verdicts", []))
            before_critiques = len(state.get("critiques", []))
            before_errors = len(state.get("errors", []))

            new_verdicts = out.get("verdicts", [])[before_verdicts:]
            # `errors` must propagate too. Without this a council that fails --
            # an LLM rate-limit under the parallel fan-out, say -- is dropped
            # silently, and the caller receives a two-council answer reporting
            # no errors. A degraded result that looks complete is worse than a
            # visible failure.
            new_errors = out.get("errors", [])[before_errors:]

            expected = len(personas_used)
            if len(new_verdicts) < expected:
                new_errors = new_errors + [
                    f"deliberation: only {len(new_verdicts)} of {expected} councils "
                    "returned a verdict"
                ]

            patch: dict[str, Any] = {
                "verdicts": new_verdicts,
                "critiques": out.get("critiques", [])[before_critiques:],
                "final_decision": out.get("final_decision", ""),
                "total_tokens": (
                    out.get("total_tokens", 0) - state.get("total_tokens", 0)
                ),
            }
            if new_errors:
                patch["errors"] = new_errors
            return patch

        builder.add_node("deliberate", deliberate_node)
        sequence.append("deliberate")

    builder.add_node("remember", memory_write_node)
    sequence.append("remember")

    builder.add_edge(START, sequence[0])
    for earlier, later in zip(sequence, sequence[1:]):
        builder.add_edge(earlier, later)
    builder.add_edge(sequence[-1], END)

    graph = builder.compile()
    graph.workflow_name = name          # type: ignore[attr-defined]
    graph.workflow_steps = sequence     # type: ignore[attr-defined]
    return graph
