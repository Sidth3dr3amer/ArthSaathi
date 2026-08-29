"""
Central configuration and path resolution.

Every module resolves paths through here rather than hardcoding them. This is the
fix for the `C:\\Users\\potda\\Namura\\...` absolute paths that used to break the
repo whenever it was copied or cloned.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ml/src/common/config.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_ROOT = PROJECT_ROOT / "ml"
DATA_DIR = ML_ROOT / "data"

# Existing on-disk assets produced by the credit-card pipeline
CARD_PIPELINE_DIR = PROJECT_ROOT / "CreditCardDataMaker_Final"
CARD_FINAL_DECISION_DIR = CARD_PIPELINE_DIR / "final_decision"
CARD_ATTRIBUTES_DIR = CARD_PIPELINE_DIR / "card_attributes"
CARD_PDF_CORPUS_DIR = CARD_PIPELINE_DIR / "per_card_data"

# Seeded reference data (created Day 1-3)
SCHEMES_FILE = DATA_DIR / "schemes.json"
TRANSACTIONS_FILE = DATA_DIR / "sample_transactions.json"

# Load .env from the project root regardless of the current working directory.
load_dotenv(PROJECT_ROOT / ".env")


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating blank strings as unset."""
    value = os.getenv(name, default)
    if value is not None and not value.strip():
        return default
    return value


def require_env(name: str) -> str:
    """Read a required environment variable or fail with an actionable message."""
    value = env(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {PROJECT_ROOT / '.env'} "
            f"(see .env.example for the expected keys)."
        )
    return value


# --------------------------------------------------------------------------- #
# Provider keys. Kept as four separate providers by explicit project decision;
# this module is only a single import point, not a consolidation.
# --------------------------------------------------------------------------- #

GROQ_API_KEY = env("GROQ_API_KEY")
CEREBRAS_API_KEY = env("CEREBRAS_API_KEY")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")
LLM7_API_KEY = env("LLM7_API_KEY")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
TAVILY_API_KEY = env("TAVILY_API_KEY")
SERPAPI_KEY = env("SERPAPI_KEY")

# Neon Postgres (memory layer)
DATABASE_URL = env("DATABASE_URL")

# Default models per provider.
#
# `llama-3.3-70b-versatile` -- what every notebook originally used -- has been
# DECOMMISSIONED by Groq and now returns 404 model_not_found. That silently
# killed every Groq path in the system (extractor, RAG answers, LLM reports,
# the router's fallback, and the anthropic->groq fallback); the agents degraded
# gracefully rather than crashing, which is exactly why it went unnoticed.
# Overridable via the GROQ_MODEL env var.
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
CEREBRAS_MODEL = env("CEREBRAS_MODEL", "gpt-oss-120b")
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-sonnet-4-6")
LLM7_MODEL = env("LLM7_MODEL", "default")
LLM7_BASE_URL = env("LLM7_BASE_URL", "https://api.llm7.io/v1/")

# OpenRouter. Chosen for the council fan-out after measuring 11 concurrent
# calls in 1.9s, against LLM7's 8.5-60s for the same deliberation.
#
# Deliberately an INSTRUCT model, not a reasoning one. Reasoning models on
# OpenRouter (openai/gpt-oss-120b, inclusionai/ling-3.0-flash-fin) spend the
# token budget on reasoning and return EMPTY content rather than an error --
# every agent would silently produce blank output while reporting success,
# which is the same failure mode as the decommissioned Groq model.
OPENROUTER_MODEL = env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_BASE_URL = env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

EMBEDDING_DIM = int(env("EMBEDDING_DIM", "384") or 384)
