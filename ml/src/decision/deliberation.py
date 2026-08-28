"""
Decision Layer -> Multi-Agent Deliberation Engine + Master Judge.

Derived from `Testing/Workflow_Testing/MultiAgent_Testing.ipynb`, which wired a
9-node LangGraph: three advisors fan out in parallel, each critiques a *different*
advisor's output (round-robin cross-criticism), and a judge synthesises the lot.

That wiring is the valuable part and is preserved exactly. What changed:

  * personas are now DATA (`Persona`), not three near-identical copy-pasted
    functions, so Day 4 can swap in the real domain councils without touching
    the graph;
  * LLM calls route through `common.llm.chat`, giving tests one seam to stub;
  * results accumulate into the unified state's `verdicts` / `critiques` lists
    instead of flat `<name>_advice` keys, which do not scale past three agents;
  * `count_tokens` no longer requires the gated Codestral tokenizer at import
    time -- it degrades to a word-count estimate when transformers is
    unavailable, so the graph runs without a HuggingFace login.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

from langgraph.graph import END, START, StateGraph

from ..common import llm
from ..schemas.state import FinancialState

DEFAULT_PROVIDER: llm.Provider = "llm7"


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #

TOKENIZER_MODEL = "mistralai/Codestral-22B-v0.1"


@lru_cache(maxsize=1)
def _tokenizer():
    """The notebook's tokenizer, loaded lazily. None if unavailable."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """
    Token count for cost accounting.

    Uses the real tokenizer when it can be loaded, otherwise a ~1.3 tokens/word
    estimate. The estimate is good enough for the cost dashboard and keeps the
    council runnable without HuggingFace credentials.
    """
    if not text:
        return 0
    tok = _tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Persona:
    """One voice in the deliberation."""

    key: str
    label: str
    brief: str
    critiques: str          # key of the persona whose advice this one critiques
    council: str = "general"
    #: Agent result keys whose findings ground this persona's argument. When set,
    #: the persona argues from computed numbers rather than from the query alone.
    result_keys: tuple[str, ...] = ()


#: Migrated from the notebook, including the round-robin critique pairing.
#: Kept as a fallback for a bare query with no council results in state.
DEFAULT_PERSONAS: tuple[Persona, ...] = (
    Persona("conservative", "Conservative", "Give low-risk advice.", critiques="growth"),
    Persona("growth", "Growth", "Give aggressive growth advice.", critiques="value"),
    Persona("value", "Value", "Give balanced investing advice.", critiques="conservative"),
)


#: The real deliberation: each council argues its own case from its agents'
#: computed findings, and critiques the council whose priorities most directly
#: compete with its own. Risk vs Growth is the central tension (protect now
#: versus compound later); Behavioral checks whether any of it will be acted on.
COUNCIL_PERSONAS: tuple[Persona, ...] = (
    Persona(
        key="risk", label="Risk Council", council="risk", critiques="growth",
        brief=(
            "You argue for protecting the household against shocks first: runway, "
            "insurance cover, and escaping high-cost debt. Argue only from the "
            "findings given. Be concrete about what breaks if a shock lands."
        ),
        result_keys=("emergency_fund_result", "insurance_result", "debt_trap_result"),
    ),
    Persona(
        key="growth", label="Growth Council", council="growth", critiques="cashflow",
        brief=(
            "You argue for compounding: time in market, retirement readiness, and "
            "the cost of delaying. Argue only from the findings given. Be concrete "
            "about what is lost by waiting another year."
        ),
        result_keys=(
            "asset_allocation_result", "retirement_result",
            "credit_card_result", "loan_advisor_result",
        ),
    ),
    Persona(
        key="cashflow", label="Cashflow Council", council="cashflow", critiques="behavioral",
        brief=(
            "You argue from what the month-to-month numbers can actually sustain: "
            "surplus, forecast stability, and whether a plan survives a bad month. "
            "Argue only from the findings given."
        ),
        result_keys=(
            "stability_result", "income_projection_result",
            "expense_optimizer_result", "goal_allocation_result",
        ),
    ),
    Persona(
        key="behavioral", label="Behavioral Council", council="behavioral", critiques="benefits",
        brief=(
            "You argue about whether this plan will actually be followed, based on "
            "observed behaviour rather than intentions. Flag any recommendation "
            "that depends on willpower the history does not support."
        ),
        result_keys=(
            "bias_detection_result", "habit_formation_result",
            "nudge_strategy_result", "literacy_result",
        ),
    ),
    Persona(
        key="benefits", label="Benefits Council", council="benefits", critiques="risk",
        brief=(
            "You argue for claiming entitlements the user already qualifies for "
            "before committing their own money. Argue only from the findings given."
        ),
        result_keys=("scheme_matching_result", "eligibility_result"),
    ),
)


def _advice_of(state: FinancialState, key: str) -> str:
    for verdict in state.get("verdicts", []):
        if verdict.get("agent") == key:
            return verdict.get("rationale", "")
    return ""


def _findings_for(state: FinancialState, persona: Persona) -> str:
    """Compact, readable summary of this council's computed findings."""
    if not persona.result_keys:
        return ""

    from ..memory.recorder import summarise      # local import avoids a cycle

    lines = []
    for key in persona.result_keys:
        result = state.get(key)
        if result:
            lines.append(f"    - {summarise(key, result)}")
    if not lines:
        return "    (this council produced no findings for this query)"
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Node factories
# --------------------------------------------------------------------------- #

def make_advisor(persona: Persona, provider: llm.Provider = DEFAULT_PROVIDER) -> Callable:
    """Build the advisor node for one persona."""

    def advisor(state: FinancialState) -> dict[str, Any]:
        findings = _findings_for(state, persona)
        findings_block = f"\n    Your council's findings:\n{findings}\n" if findings else ""
        prompt = f"""
    User Query:
    {state.get('query', '')}
{findings_block}
    {persona.brief}
    """
        try:
            answer = llm.chat(prompt, provider=provider)
        except Exception as exc:
            return {"errors": [f"{persona.key}_advisor: {exc!r}"]}

        tokens = count_tokens(prompt) + count_tokens(answer)
        return {
            "verdicts": [{
                "agent": persona.key,
                "council": persona.council,
                "stance": persona.label,
                "rationale": answer,
                "confidence": 0.0,
                "tokens": tokens,
            }],
            "total_tokens": tokens,
        }

    advisor.__name__ = f"{persona.key}_advisor"
    return advisor


def make_critic(persona: Persona, provider: llm.Provider = DEFAULT_PROVIDER) -> Callable:
    """Build the critic node for one persona (it critiques `persona.critiques`)."""

    def critic(state: FinancialState) -> dict[str, Any]:
        target_advice = _advice_of(state, persona.critiques)
        if not target_advice:
            return {"errors": [f"{persona.key}_critic: no {persona.critiques} advice to critique"]}

        prompt = f"""
    Critique this {persona.critiques} strategy:

    {target_advice}
    """
        try:
            answer = llm.chat(prompt, provider=provider)
        except Exception as exc:
            return {"errors": [f"{persona.key}_critic: {exc!r}"]}

        tokens = count_tokens(prompt) + count_tokens(answer)
        return {
            "critiques": [{
                "agent": persona.key,
                "council": persona.council,
                "stance": f"critique_of_{persona.critiques}",
                "rationale": answer,
                "tokens": tokens,
            }],
            "total_tokens": tokens,
        }

    critic.__name__ = f"{persona.key}_critic"
    return critic


def make_judge(provider: llm.Provider = DEFAULT_PROVIDER) -> Callable:
    """Build the Master Judge node: synthesises every verdict and critique."""

    def judge(state: FinancialState) -> dict[str, Any]:
        verdicts = state.get("verdicts", [])
        critiques = state.get("critiques", [])
        if not verdicts:
            return {"final_decision": "", "errors": ["judge: no verdicts to synthesise"]}

        sections = [f"    QUERY:\n    {state.get('query', '')}\n"]
        for v in verdicts:
            sections.append(f"    {v['stance'].upper()}:\n    {v['rationale']}\n")
        for c in critiques:
            sections.append(f"    {c['stance'].upper()}:\n    {c['rationale']}\n")
        sections.append("    Produce final recommendation.")
        prompt = "\n".join(sections)

        try:
            answer = llm.chat(prompt, provider=provider)
        except Exception as exc:
            return {"final_decision": "", "errors": [f"judge: {exc!r}"]}

        tokens = count_tokens(prompt) + count_tokens(answer)
        return {"final_decision": answer, "total_tokens": tokens}

    judge.__name__ = "judge"
    return judge


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def build_deliberation_graph(
    personas: Sequence[Persona] = DEFAULT_PERSONAS,
    provider: llm.Provider = DEFAULT_PROVIDER,
):
    """
    Wire the deliberation graph.

    Topology, preserved from the notebook:

        START -> every advisor (parallel fan-out)
        advisor(X) -> the critic whose `critiques` field names X
        every critic -> judge -> END
    """
    if not personas:
        raise ValueError("at least one persona is required")

    keys = {p.key for p in personas}
    for p in personas:
        if p.critiques not in keys:
            raise ValueError(
                f"persona {p.key!r} critiques {p.critiques!r}, which is not in the set"
            )

    builder = StateGraph(FinancialState)

    for p in personas:
        builder.add_node(p.key, make_advisor(p, provider))
        builder.add_node(f"{p.key}_critic", make_critic(p, provider))
    builder.add_node("judge", make_judge(provider))

    for p in personas:
        builder.add_edge(START, p.key)
        # the critic depends on the advice it critiques, so it runs after that advisor
        builder.add_edge(p.critiques, f"{p.key}_critic")
        builder.add_edge(f"{p.key}_critic", "judge")

    builder.add_edge("judge", END)
    return builder.compile()


def deliberate(
    state: FinancialState,
    personas: Sequence[Persona] = DEFAULT_PERSONAS,
    provider: llm.Provider = DEFAULT_PROVIDER,
) -> FinancialState:
    """Run a full deliberation and return the resulting state."""
    return build_deliberation_graph(personas, provider).invoke(state)
