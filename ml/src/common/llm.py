"""
LLM access for every agent.

All four providers are kept, by explicit project decision — this module does NOT
consolidate which provider an agent uses. It exists so that:

  * clients are created lazily (importing an agent never requires an API key), and
  * tests have a single seam to monkeypatch instead of four.

Provider assignments inherited from the original notebooks:

  groq      -> most council agents, fraud detection, voice, query funnel
  cerebras  -> credit-card PDF attribute extraction
  anthropic -> cashflow narrative layer
  llm7      -> multi-agent deliberation council (OpenAI-compatible endpoint)

Agents should call `chat(prompt, provider="groq")` rather than constructing clients.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from . import config

Provider = Literal["groq", "cerebras", "anthropic", "llm7"]


class LLMNotConfigured(RuntimeError):
    """Raised when an agent needs a provider whose API key is absent."""


# --------------------------------------------------------------------------- #
# Lazy clients
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def groq_client():
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise LLMNotConfigured("GROQ_API_KEY is not set.")
    return Groq(api_key=config.GROQ_API_KEY)


@lru_cache(maxsize=1)
def cerebras_client():
    from cerebras.cloud.sdk import Cerebras

    if not config.CEREBRAS_API_KEY:
        raise LLMNotConfigured("CEREBRAS_API_KEY is not set.")
    return Cerebras(api_key=config.CEREBRAS_API_KEY)


@lru_cache(maxsize=1)
def anthropic_client():
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise LLMNotConfigured(
            "ANTHROPIC_API_KEY is not set. The cashflow narrative layer needs it; "
            "the deterministic forecast/simulation core runs without it."
        )
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def llm7_client():
    from openai import OpenAI

    if not config.LLM7_API_KEY:
        raise LLMNotConfigured("LLM7_API_KEY is not set.")
    return OpenAI(base_url=config.LLM7_BASE_URL, api_key=config.LLM7_API_KEY)


_DEFAULT_MODEL: dict[str, str] = {
    "groq": config.GROQ_MODEL,
    "cerebras": config.CEREBRAS_MODEL,
    "anthropic": config.ANTHROPIC_MODEL,
    "llm7": config.LLM7_MODEL,
}

_KEY_FOR: dict[str, str | None] = {
    "groq": config.GROQ_API_KEY,
    "cerebras": config.CEREBRAS_API_KEY,
    "anthropic": config.ANTHROPIC_API_KEY,
    "llm7": config.LLM7_API_KEY,
}

#: Where to route when the requested provider has no credentials.
#: ANTHROPIC_API_KEY is currently a blank placeholder, so the cashflow narrative
#: layer runs on Groq until a real key is supplied -- no code change needed then,
#: it simply stops falling back.
PROVIDER_FALLBACK: dict[str, Provider] = {
    "anthropic": "groq",
    "llm7": "groq",
    "cerebras": "groq",
}


def is_configured(provider: Provider) -> bool:
    return bool(_KEY_FOR.get(provider))


def resolve_provider(provider: Provider, *, strict: bool = False) -> Provider:
    """
    Return the provider to actually call.

    Falls back per `PROVIDER_FALLBACK` when the requested one has no key, so a
    missing optional credential degrades instead of breaking a workflow. Pass
    `strict=True` to demand the exact provider.
    """
    if is_configured(provider):
        return provider
    if strict:
        raise LLMNotConfigured(f"{provider.upper()}_API_KEY is not set.")

    fallback = PROVIDER_FALLBACK.get(provider)
    if fallback and is_configured(fallback):
        return fallback
    raise LLMNotConfigured(
        f"{provider.upper()}_API_KEY is not set and no configured fallback is "
        f"available (tried {fallback!r})."
    )


# --------------------------------------------------------------------------- #
# Unified call surface
# --------------------------------------------------------------------------- #

def chat(
    prompt: str,
    *,
    provider: Provider = "groq",
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    strict: bool = False,
    **kwargs: Any,
) -> str:
    """
    Send a single-turn prompt and return the text response.

    If `provider` has no credentials it degrades via `PROVIDER_FALLBACK` unless
    `strict=True`. This is the seam tests monkeypatch:

        monkeypatch.setattr("ml.src.common.llm.chat", lambda *a, **k: "stub")
    """
    requested = provider
    provider = resolve_provider(provider, strict=strict)
    # A caller's model name belongs to the provider they asked for; drop it when
    # we have been rerouted, so we do not send e.g. a Claude id to Groq.
    if provider != requested:
        model = None
    model = model or _DEFAULT_MODEL[provider]

    if provider == "anthropic":
        client = anthropic_client()
        message = client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 2048),
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in message.content if block.type == "text")

    # groq / cerebras / llm7 all expose an OpenAI-compatible chat.completions API
    client = {
        "groq": groq_client,
        "cerebras": cerebras_client,
        "llm7": llm7_client,
    }[provider]()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def available_providers() -> dict[str, bool]:
    """Which providers currently have credentials. Useful in demos and diagnostics."""
    return {
        "groq": bool(config.GROQ_API_KEY),
        "cerebras": bool(config.CEREBRAS_API_KEY),
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "llm7": bool(config.LLM7_API_KEY),
    }
