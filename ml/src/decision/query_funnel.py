"""
Decision Layer -> Query Funnel.

Migrated verbatim from `TestVoice/query_funnel_pipeline.ipynb`.

Turns a natural-language question into a structured plan and executes it as a
funnel over a list of records:

    discover_schema  -> infer available fields from the data
    plan_query       -> LLM turns the question into {domains, fields, filter, computation}
    filter_by_domain -> narrow to relevant records
    project_fields   -> keep only the requested fields
    apply_record_filter -> apply comparison predicates
    compute_aggregate   -> sum / avg / min / max / count
    run_funnel       -> the four steps above, in order
    ask              -> plan + run + explain, end to end

Only `plan_query`, `explain` and `ask` touch an LLM; the funnel stages are pure
list/dict transforms and are unit-tested directly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from ..common import config, llm

PLANNER_SYSTEM_PROMPT = """
You are a query-planning agent.

You will be given:

1. A USER QUESTION
2. The AVAILABLE SCHEMA of a dataset

You NEVER see actual records.
You NEVER answer the question.
Your only job is to produce a retrieval/computation plan.

AVAILABLE SCHEMA:
{schema}

--------------------------------------------------
CLASSIFICATION RULES
--------------------------------------------------

Set "is_specific" = true if answering the question requires
looking at dataset records.

Examples:

- Which strategy is best?
- Which option is safest?
- Which strategy is more aggressive?
- Which one should I choose?
- Which strategy has highest returns?
- Which plan has lowest risk?
- Compare the available strategies.
- Rank the strategies.
- Which option is good for retirement?
- Which option is balanced?
- I want a safe option.
- I don't want a lot of headache.
- I want stable returns.
- Which one is most suitable for me?

These ALL require dataset analysis and MUST return:

"is_specific": true

--------------------------------------------------

Set "is_specific" = false ONLY when the question can be
answered without any dataset records.

Examples:

- What is a mutual fund?
- What is volatility?
- Explain drawdown.
- What is asset allocation?
- What does CAGR mean?
- Explain retirement planning.

--------------------------------------------------
PLANNING RULES
--------------------------------------------------

If is_specific = true:

Return:

- domains
- fields
- record_filter
- computation
- computation_field

Rules:

1. Select ONLY fields needed to answer the question.

2. Use:

"record_filter": "all records"

when the user is comparing or choosing among options.

3. Use:

"computation": "compare"

for questions involving:

- best
- safest
- recommend
- choose
- rank
- compare
- aggressive
- conservative
- balanced
- suitable
- highest
- lowest

4. Use:

"max" for explicit highest-value requests.

Example:
"Which strategy has highest expected return?"

5. Use:

"min" for explicit lowest-value requests.

Example:
"Which strategy has lowest volatility?"

6. NEVER invent field names.

Only use fields that exist in the schema.

7. NEVER treat field names as domains.

Domains are logical dataset groups.
Fields are attributes inside domains.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return STRICT JSON ONLY.

{{
  "is_specific": true,
  "reasoning": "...",
  "domains": [...],
  "fields": [...],
  "record_filter": "...",
  "computation": "...",
  "computation_field": "..."
}}

OR

{{
  "is_specific": false,
  "reasoning": "...",
  "domains": null,
  "fields": null,
  "record_filter": null,
  "computation": null,
  "computation_field": null
}}

No markdown.
No explanations.
No prose outside JSON.
"""


_COMPARATORS = {
    "less than": "<", "under": "<", "below": "<",
    "more than": ">", "greater than": ">", "above": ">", "over": ">",
    "equals": "==", "equal to": "==", "is": "==",
    "at least": ">=", "at most": "<=",
}


EXPLAINER_SYSTEM_GENERAL = """You are a helpful financial assistant. Answer the user's
question directly using your general knowledge. This question does not require looking up
any specific dataset -- answer it the way you normally would."""


EXPLAINER_SYSTEM_SPECIFIC = """You are a data-grounded financial assistant. You have been
given a NARROWED SLICE of records (already filtered to what's relevant) and, if applicable,
a CODE-COMPUTED AGGREGATE (already calculated deterministically -- treat this number as
authoritative, do not recompute or second-guess it; just explain it).

NARROWED RECORDS:
{records}

COMPUTED AGGREGATE (if any):
{aggregate}

Answer the user's question using ONLY this data. Rules:
- If an aggregate is provided, state it clearly and explain what it means in context.
- Reference specific records (row_id or relevant field values) when useful for the answer.
- If the narrowed records don't contain enough information to fully answer, say so explicitly.
- Be concise and direct. Lead with the answer.
"""
def discover_schema(strategies):
    schema = {}

    for strategy in strategies:

        for key, value in strategy.items():

            if isinstance(value, dict):

                schema[key] = sorted(value.keys())

            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], dict)
            ):

                fields = set()

                for item in value:
                    fields.update(item.keys())

                schema[key] = sorted(fields)

            else:

                schema[key] = type(value).__name__

    return schema


def plan_query(
    question: str,
    schema: dict,
    model: str | None = None,
) -> dict:
    # `llama-3.3-70b-versatile` was hardcoded here by the original notebook and
    # has since been decommissioned by Groq. Default to the configured model.
    model = model or config.GROQ_MODEL

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        schema=json.dumps(schema, indent=2)
    )

    try:

        completion = llm.groq_client().chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        text = completion.choices[0].message.content.strip()

        plan = json.loads(text)

    except Exception as e:

        print(f"⚠️ Planning failed: {e}")

        plan = {
            "is_specific": True,
            "reasoning": "fallback: planner call failed",
            "domains": list(schema.keys()),
            "fields": None,
            "record_filter": "all records",
            "computation": "none",
            "computation_field": None,
        }

    return plan


def filter_by_domain(rows: list[dict], domains: list[str]) -> list[dict]:
    """
    Safe domain filtering.

    If planner returns invalid domains (e.g. field names instead of domains),
    do NOT drop all rows.
    """

    if not domains:
        return rows

    available_domains = {r.get("domain") for r in rows}

    valid_domains = [
        d for d in domains
        if d in available_domains
    ]

    if not valid_domains:
        print(
            f"⚠️ Planner returned invalid domains {domains}. "
            f"Available domains: {available_domains}. "
            f"Skipping domain filter."
        )
        return rows

    return [
        r for r in rows
        if r.get("domain") in valid_domains
    ]


def project_fields(rows: list[dict], fields: list[str]) -> list[dict]:
    """Keep row_id + domain always (for traceability) plus only the requested fields."""
    if not fields:
        return rows
    return [
        {k: r[k] for k in ("row_id", "domain", *fields) if k in r}
        for r in rows
    ]


def apply_record_filter(rows: list[dict], filter_desc: str) -> list[dict]:
    """
    Lightweight mechanical parser for simple comparison filters like
    '<field> less than <value>' or '<field> equals <value>'.
    If the filter can't be confidently parsed, returns ALL rows unchanged
    (safe default -- never silently drops data based on a guess).
    """
    if not filter_desc or filter_desc.strip().lower() in ("all records", "none", "no filter"):
        return rows

    desc = filter_desc.lower()
    for phrase, op in _COMPARATORS.items():
        if phrase in desc:
            # crude field/value extraction: "<field> <phrase> <value>"
            parts = desc.split(phrase)
            if len(parts) != 2:
                continue
            field_part = parts[0].strip().replace(" ", "_")
            value_part = parts[1].strip().split()[0].strip(".,")

            # find the actual field name (allow partial match against row keys)
            sample = rows[0] if rows else {}
            matched_field = next(
                (k for k in sample.keys() if k.lower() in field_part or field_part in k.lower()),
                None,
            )
            if not matched_field:
                continue

            try:
                value = float(value_part)
            except ValueError:
                value = value_part.strip('"\'')

            def cmp(row_val, op=op, value=value):
                try:
                    rv = float(row_val)
                    if op == "<":  return rv < value
                    if op == ">":  return rv > value
                    if op == ">=": return rv >= value
                    if op == "<=": return rv <= value
                    if op == "==": return rv == value
                except (TypeError, ValueError):
                    if op == "==":
                        return str(row_val).lower() == str(value).lower()
                return False

            filtered = [r for r in rows if matched_field in r and cmp(r[matched_field])]
            return filtered

    # Could not parse -- safe fallback, include everything and let the
    # explainer reason over the unfiltered set rather than silently dropping rows.
    return rows


def compute_aggregate(rows: list[dict], computation: str, field: str) -> Optional[dict]:
    """Deterministic, code-level computation -- no LLM, no rounding surprises."""
    if not computation or computation == "none" or not field:
        return None

    values = []
    for r in rows:
        if field in r:
            try:
                values.append(float(r[field]))
            except (TypeError, ValueError):
                continue

    if not values:
        return {"computation": computation, "field": field, "result": None, "n": 0}

    if computation == "sum":      result = sum(values)
    elif computation == "average":result = sum(values) / len(values)
    elif computation == "count":  result = len(values)
    elif computation == "max":    result = max(values)
    elif computation == "min":    result = min(values)
    else:                         result = None   # "compare" handled by explainer over raw rows

    return {"computation": computation, "field": field, "result": result, "n": len(values)}


def run_funnel(rows: list[dict], plan: dict) -> dict:
    """
    Executes:
    domain -> record filter -> field projection -> aggregate
    """

    if not plan.get("is_specific"):
        return {
            "narrowed_rows": [],
            "aggregate": None,
            "skipped": True
        }

    domains = plan.get("domains") or []
    fields = plan.get("fields") or []
    record_filter = plan.get("record_filter") or "all records"
    computation = plan.get("computation") or "none"
    computation_field = plan.get("computation_field")

    print("\nDEBUG")
    print("Planner domains:", domains)
    print("Available domains:", {r.get("domain") for r in rows})

    step1 = filter_by_domain(rows, domains)

    print("Rows after domain filter:", len(step1))

    step2 = apply_record_filter(step1, record_filter)
    step3 = project_fields(step2, fields)

    aggregate = compute_aggregate(
        step2,
        computation,
        computation_field
    )

    return {
        "narrowed_rows": step3,
        "aggregate": aggregate,
        "skipped": False,
        "funnel_trace": {
            "after_domain_filter": len(step1),
            "after_record_filter": len(step2),
            "after_field_projection": len(step3),
        },
    }


def explain(
    question: str,
    plan: dict,
    funnel_result: dict,
    model: str | None = None,
    max_tokens: int = 700,
) -> str:
    model = model or config.GROQ_MODEL

    if funnel_result["skipped"]:
        system_prompt = EXPLAINER_SYSTEM_GENERAL
    else:
        system_prompt = EXPLAINER_SYSTEM_SPECIFIC.format(
            records=json.dumps(
                funnel_result["narrowed_rows"],
                indent=2
            ),
            aggregate=(
                json.dumps(
                    funnel_result["aggregate"],
                    indent=2
                )
                if funnel_result["aggregate"]
                else "none"
            ),
        )

    try:

        completion = llm.groq_client().chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"⚠️ Explainer API error: {e}"


def ask(
    question: str,
    rows: list[dict] = None,
    schema: dict = None,
    verbose: bool = True
) -> dict:

    rows = rows if rows is not None else strategies
    strategies = strategies if strategies is not None else []
    schema = schema if schema is not None else discover_schema(strategies)

    # Convert strategies into funnel-compatible rows
    if rows and isinstance(rows[0], dict) and "domain" not in rows[0]:
        rows = [
            {
                "row_id": i,
                "domain": "strategy",
                **r
            }
            for i, r in enumerate(rows)
        ]

    if verbose:
        print(f"❓ Question: {question}\n")
        print("🧭 Stage 1 — Planning (question only, no data shown)...")

    plan = plan_query(question, schema)

    if verbose:
        print(f"   is_specific: {plan.get('is_specific')}")
        print(f"   reasoning:   {plan.get('reasoning', '')}")

        if plan.get("is_specific"):
            print(f"   domains:     {plan.get('domains')}")
            print(f"   fields:      {plan.get('fields')}")
            print(f"   filter:      {plan.get('record_filter')}")
            print(
                f"   computation: {plan.get('computation')} "
                f"on {plan.get('computation_field')}"
            )

    if plan.get("is_specific"):

        if verbose:
            print(
                "\n🔻 Stage 2 — Narrowing funnel "
                "(domain → fields → records → compute)..."
            )

        funnel_result = run_funnel(rows, plan)

        if verbose:
            t = funnel_result["funnel_trace"]

            print(
                f"   {len(rows)} rows → domain filter → "
                f"{t['after_domain_filter']} "
                f"→ record filter → {t['after_record_filter']} "
                f"→ field projection → "
                f"{t['after_field_projection']} rows"
            )

            if funnel_result["aggregate"]:
                print(f"   Computed: {funnel_result['aggregate']}")

    else:

        if verbose:
            print(
                "\n⏭️ Stage 2 — Skipped "
                "(general question, no data needed)"
            )

        funnel_result = {
            "narrowed_rows": [],
            "aggregate": None,
            "skipped": True
        }

    if verbose:
        print("\n💬 Stage 3 — Explaining...\n")

    answer = explain(question, plan, funnel_result)

    return {
        "question": question,
        "plan": plan,
        "funnel_result": funnel_result,
        "answer": answer
    }
