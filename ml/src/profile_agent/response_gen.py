"""
Profile Agent -> Response Generator.

Produces the reply the user sees, and runs the whole slide-8 pipeline end to end.

Three behaviours worth naming:

  * **It confirms what it recorded.** Every applied change is echoed back, so a
    misheard income is corrected in the next breath rather than silently
    poisoning six councils.
  * **It surfaces held-back changes as questions.** The updater refuses
    suspicious writes; this is where the user gets asked about them.
  * **It replies in the user's language.** The Input Processor detects the
    script, and the system prompt carries it through -- matching the deck's
    multilingual promise without a separate translation step.

A deterministic reply is always constructed first. The LLM only rewrites it into
natural prose, so a provider outage degrades to a plainer answer rather than no
answer.
"""

from __future__ import annotations

from typing import Any

from ..common import llm
from ..memory.store import MemoryStore
from ..schemas.profile import UserProfile
from . import memory_creator, question_gen, rag, updater
from .extractor import extract_information
from .input_processor import process_input

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "kn": "Kannada",
    "ta": "Tamil", "bn": "Bengali",
}

RESPONSE_SYSTEM = (
    "You are Artha, a warm and plain-spoken financial assistant for Indian users. "
    "Rewrite the draft below into a natural reply. Rules:\n"
    "- Reply ONLY in {language}.\n"
    "- Keep every number exactly as given. Never invent a figure.\n"
    "- Keep it under 90 words.\n"
    "- If a question is included, end with it.\n"
    "- No jargon, no financial advice beyond what the draft states."
)


def compose_draft(
    merged: dict[str, Any],
    next_question: dict[str, Any] | None,
    processed: dict[str, Any],
) -> str:
    """Build the deterministic reply. This is what the user gets if the LLM is down."""
    parts: list[str] = []

    if merged.get("applied"):
        noted = ", ".join(
            f"{c['field'].replace('_', ' ')} as "
            + (f"Rs {c['to']:,.0f}" if isinstance(c["to"], (int, float))
               and c["field"] not in ("age", "dependents") else str(c["to"]))
            for c in merged["applied"]
        )
        parts.append(f"Noted — I've recorded your {noted}.")
    elif not processed.get("is_empty"):
        parts.append("Thanks.")

    for pending in merged.get("needs_confirmation", []):
        parts.append(
            f"You mentioned {pending['field'].replace('_', ' ')} as "
            f"{pending['proposed']}, but I had {pending['current']} on record. "
            "Which is right?"
        )

    if next_question:
        parts.append(next_question["why_we_ask"])
        parts.append(next_question["question"])
    elif not merged.get("needs_confirmation"):
        parts.append(
            "I have enough to work with now — ask me anything about your finances."
        )

    return " ".join(parts)


def generate_response(
    draft: str,
    language: str = "en",
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """Rewrite the draft naturally. Falls back to the draft on any failure."""
    if not draft.strip():
        return {"response": "", "method": "empty"}

    system = RESPONSE_SYSTEM.format(
        language=LANGUAGE_NAMES.get(language, "English")
    )
    try:
        text = llm.chat(draft, provider=provider, system=system)
    except Exception as exc:
        return {"response": draft, "method": "fallback_draft", "error": repr(exc)}

    return {
        "response": (text or "").strip() or draft,
        "method": "llm",
        "draft": draft,
    }


def run_profile_agent(
    user_message: str,
    profile: UserProfile,
    store: MemoryStore | None = None,
    provider: llm.Provider = "groq",
    persist: bool = True,
) -> dict[str, Any]:
    """
    The complete slide-8 pipeline:

        InputProcessor -> InformationExtractor -> ProfileUpdater
                       -> MemoryCreator -> QuestionGenerator -> ResponseGenerator

    Returns the updated profile alongside every stage's output, so a UI can show
    progress and a test can assert on any stage.
    """
    processed = process_input(user_message)
    extraction = extract_information(processed, provider=provider)
    merged = updater.update_profile(profile, extraction, store=store, persist=persist)
    updated: UserProfile = merged["profile"]

    memories = {"memories_written": 0}
    if persist and (merged["applied"] or user_message.strip()):
        memories = memory_creator.create_memories(
            user_id=updated.user_id,
            applied=merged["applied"],
            user_message=user_message,
            store=store,
        )

    plan = question_gen.question_plan(updated)
    draft = compose_draft(merged, plan["next_question"], processed)
    response = generate_response(draft, processed["language"], provider=provider)

    return {
        "profile": updated,
        "response": response["response"],
        "stages": {
            "input_processor": processed,
            "extractor": extraction,
            "updater": {k: v for k, v in merged.items() if k != "profile"},
            "memory_creator": memories,
            "question_generator": plan,
            "response_generator": {k: v for k, v in response.items() if k != "draft"},
        },
        "next_question": plan["next_question"],
        "completeness": plan["completeness"],
        "needs_confirmation": merged["needs_confirmation"],
    }


def answer_with_context(
    question: str,
    profile: UserProfile,
    store: MemoryStore | None = None,
    provider: llm.Provider = "groq",
) -> dict[str, Any]:
    """
    Answer a question using retrieved profile and memory -- the RAG read path,
    as opposed to `run_profile_agent`, which is the write path.
    """
    processed = process_input(question)
    built = rag.retrieve_and_build(profile, processed["cleaned"], store=store)

    system = (
        "You are Artha, a financial assistant for Indian users. Answer using ONLY "
        "the context provided. If the context does not contain the answer, say so "
        "plainly and name what you would need to know. Reply in "
        f"{LANGUAGE_NAMES.get(processed['language'], 'English')}, under 120 words."
    )
    prompt = f"{built['context']}\n\nQUESTION\n{processed['cleaned']}"

    try:
        answer = llm.chat(prompt, provider=provider, system=system)
        method = "llm"
    except Exception as exc:
        answer = (
            "I cannot reach the language service right now. Here is what I have "
            f"on record:\n\n{built['profile_summary']}"
        )
        method = f"fallback: {exc!r}"

    return {
        "answer": answer,
        "method": method,
        "context_used": built["context"],
        "memories_included": built["memories_included"],
        "language": processed["language"],
    }
