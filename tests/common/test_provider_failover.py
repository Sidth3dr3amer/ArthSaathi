"""
Provider failover.

Groq serves roughly nine LLM calls in ten. Before these behaviours existed it
was the one provider nothing fell back *from*, and fallback only engaged on a
missing key — never on a live outage — so a provider whose key was present but
whose service was down took the whole system with it.
"""

from __future__ import annotations

import pytest

import ml.src.common.llm as llm


@pytest.fixture
def dead(monkeypatch):
    """Patch `_call` so the named providers fail as if the service were down."""

    def _install(*down: str):
        def fake(prompt, provider, system, model, temperature, **kwargs):
            if provider in down:
                raise RuntimeError(f"Error code: 503 - {provider} unavailable")
            return f"served by {provider}"

        monkeypatch.setattr(llm, "_call", fake)

    return _install


# --------------------------------------------------------------------------- #
# The fallback map
# --------------------------------------------------------------------------- #

def test_groq_has_a_fallback():
    """Everything falls back to Groq; Groq must fall back somewhere too."""
    assert llm.PROVIDER_FALLBACK.get("groq")


def test_every_fallback_target_is_a_real_provider():
    valid = set(llm._DEFAULT_MODEL)
    for source, target in llm.PROVIDER_FALLBACK.items():
        assert source in valid, source
        assert target in valid, target
        assert source != target, f"{source} cannot fall back to itself"


# --------------------------------------------------------------------------- #
# Missing credentials
# --------------------------------------------------------------------------- #

def test_absent_key_resolves_to_the_fallback(monkeypatch):
    monkeypatch.setitem(llm._KEY_FOR, "anthropic", None)
    monkeypatch.setitem(llm._KEY_FOR, "groq", "present")
    assert llm.resolve_provider("anthropic") == "groq"


def test_strict_refuses_to_reroute_an_absent_key(monkeypatch):
    monkeypatch.setitem(llm._KEY_FOR, "anthropic", None)
    with pytest.raises(llm.LLMNotConfigured):
        llm.resolve_provider("anthropic", strict=True)


def test_absent_key_with_absent_fallback_raises(monkeypatch):
    monkeypatch.setitem(llm._KEY_FOR, "anthropic", None)
    monkeypatch.setitem(llm._KEY_FOR, "groq", None)
    with pytest.raises(llm.LLMNotConfigured):
        llm.resolve_provider("anthropic")


# --------------------------------------------------------------------------- #
# Live outages — the gap this file exists to close
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("provider", ["groq", "llm7", "anthropic", "cerebras"])
def test_a_live_outage_falls_over(dead, provider):
    dead(provider)
    result = llm.chat("hi", provider=provider)
    assert result != f"served by {provider}"
    assert result.startswith("served by ")


def test_a_healthy_provider_is_never_rerouted(dead):
    dead()
    assert llm.chat("hi", provider="groq") == "served by groq"


def test_both_providers_down_reports_the_original_failure(dead):
    """The caller needs to hear about the provider it asked for."""
    fallback = llm.PROVIDER_FALLBACK["groq"]
    dead("groq", fallback)
    with pytest.raises(RuntimeError, match="groq unavailable"):
        llm.chat("hi", provider="groq")


def test_failover_does_not_loop(dead):
    """A -> B -> A would otherwise recurse."""
    fallback = llm.PROVIDER_FALLBACK["groq"]
    dead("groq", fallback)
    with pytest.raises(RuntimeError):
        llm.chat("hi", provider="groq")      # terminates rather than hanging


def test_strict_refuses_to_reroute_a_live_outage(dead):
    dead("groq")
    with pytest.raises(RuntimeError, match="groq unavailable"):
        llm.chat("hi", provider="groq", strict=True)


def test_failover_does_not_carry_the_original_model(monkeypatch):
    """A Claude model id must never be sent to Groq."""
    seen = []

    def fake(prompt, provider, system, model, temperature, **kwargs):
        seen.append((provider, model))
        if provider == "anthropic":
            raise RuntimeError("Error code: 503 - anthropic unavailable")
        return "ok"

    monkeypatch.setattr(llm, "_call", fake)
    monkeypatch.setitem(llm._KEY_FOR, "anthropic", "present")
    llm.chat("hi", provider="anthropic", model="claude-sonnet-4-6")

    assert seen[0] == ("anthropic", "claude-sonnet-4-6")
    assert seen[1][0] == "groq"
    assert seen[1][1] == llm._DEFAULT_MODEL["groq"]


def test_rate_limit_is_retried_before_failing_over(monkeypatch):
    """A 429 is transient; retry the same provider rather than rerouting."""
    attempts = []

    def fake(prompt, provider, system, model, temperature, **kwargs):
        attempts.append(provider)
        if provider == "groq" and len(attempts) < 2:
            raise RuntimeError("Error code: 429 - Rate limit. Retry after 0.01 seconds.")
        return f"served by {provider}"

    monkeypatch.setattr(llm, "_call", fake)
    monkeypatch.setattr(llm, "MAX_BACKOFF_SECONDS", 0.05)
    assert llm.chat("hi", provider="groq") == "served by groq"
    assert attempts == ["groq", "groq"]


def test_a_missing_key_error_is_not_swallowed_by_failover(monkeypatch):
    """With no key anywhere, the caller gets LLMNotConfigured, not a live error."""
    for name in llm._KEY_FOR:
        monkeypatch.setitem(llm._KEY_FOR, name, None)
    with pytest.raises(llm.LLMNotConfigured):
        llm.chat("hi", provider="groq")


def test_openrouter_is_wired_as_a_provider():
    """Added as Groq's failover target: a different vendor, independently keyed."""
    assert "openrouter" in llm._KEY_FOR
    assert "openrouter" in llm._DEFAULT_MODEL
    assert llm.PROVIDER_FALLBACK["groq"] == "openrouter"


def test_deliberation_does_not_use_a_reasoning_model():
    """
    Reasoning models on OpenRouter (gpt-oss-120b, ling-3.0-flash-fin) spend the
    token budget on reasoning and return EMPTY content rather than an error --
    every council would go blank while reporting success.
    """
    assert "gpt-oss" not in llm._DEFAULT_MODEL["openrouter"]
    assert "ling" not in llm._DEFAULT_MODEL["openrouter"]
    assert "instruct" in llm._DEFAULT_MODEL["openrouter"].lower()
