"""
Guard against a configured model id going stale.

Groq decommissioned `llama-3.3-70b-versatile` -- the id every notebook used and
which `config.GROQ_MODEL` still defaulted to. Every Groq path in the system
(profile extraction, RAG answers, LLM reports, the router's fallback, and the
anthropic->groq fallback) returned 404 model_not_found.

It went unnoticed precisely because the agents degrade gracefully: the RAG
answer returned a profile summary, the extractor returned no fields, the router
fell back to `general`. Nothing crashed, so nothing complained.

The offline tests below pin the configuration itself; the `live` test asks the
provider whether the id still exists and is the one that would actually catch a
future decommissioning.
"""

from __future__ import annotations

import pytest

from ml.src.common import config, llm

#: Model ids known to be retired. A configured default must never be one of these.
DECOMMISSIONED = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
}


def test_no_provider_defaults_to_a_decommissioned_model():
    for name, model in (
        ("GROQ_MODEL", config.GROQ_MODEL),
        ("CEREBRAS_MODEL", config.CEREBRAS_MODEL),
        ("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL),
    ):
        assert model not in DECOMMISSIONED, (
            f"{name} is set to {model!r}, which has been retired by the provider"
        )


def test_every_provider_has_a_default_model():
    for provider in ("groq", "cerebras", "anthropic", "llm7"):
        assert llm._DEFAULT_MODEL[provider], provider


def test_no_module_hardcodes_a_decommissioned_model():
    """
    Model ids belong in `config`, not scattered through agent modules -- two
    were hardcoded in `query_funnel` and survived the decommissioning unnoticed.
    """
    import pathlib

    offenders: list[str] = []
    for path in pathlib.Path("ml/src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue                       # explanatory comments are fine
            if any(dead in stripped for dead in DECOMMISSIONED):
                offenders.append(f"{path}: {stripped[:70]}")
    assert not offenders, "decommissioned model id hardcoded:\n" + "\n".join(offenders)


@pytest.mark.live
def test_configured_groq_model_still_exists():
    """Asks Groq directly. This is the test that catches the next decommissioning."""
    if not config.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not configured")

    available = {m.id for m in llm.groq_client().models.list().data}
    assert config.GROQ_MODEL in available, (
        f"GROQ_MODEL={config.GROQ_MODEL!r} is not offered by the account. "
        f"Available: {sorted(available)}"
    )


@pytest.mark.live
def test_configured_groq_model_answers():
    if not config.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not configured")
    assert llm.chat("Reply with exactly: OK", provider="groq").strip()
