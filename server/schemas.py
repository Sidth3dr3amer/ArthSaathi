"""
Request and response models for the API.

`UserProfile` is imported, never redefined -- it is the contract every council
agent already reads, and a second copy here would drift within a week.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ml.src.schemas.profile import UserProfile


class ChatRequest(BaseModel):
    """The main entry point: a message from a user."""

    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4_000)
    #: Optional inline profile. Overrides the stored one when supplied.
    profile: UserProfile | None = None
    #: Optional transaction history for the Behavioral Council. Omitted means
    #: the seeded demo dataset is used.
    transactions: list[dict[str, Any]] | None = None
    #: Routing falls back to an LLM only when the deterministic rules miss.
    use_llm_router: bool = True


class ChatResponse(BaseModel):
    intent: str
    workflow: str
    agents_run: list[str]
    recommendations: dict[str, list[str]]
    allocation_plan: list[dict[str, Any]]
    final_decision: str
    council_verdicts: list[dict[str, Any]]
    memory_written: bool
    errors: list[str]
    elapsed_seconds: float
    #: Where the profile came from: inline, stored, or new. Must be declared --
    #: `response_model` silently drops any field not on the model.
    profile_source: str = "new"


class ProfileTurnRequest(BaseModel):
    """One turn of the Profile Agent onboarding conversation."""

    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4_000)
    profile: UserProfile | None = None
    persist: bool = True


class ProfileTurnResponse(BaseModel):
    user_id: str
    response: str
    profile: dict[str, Any]
    next_question: dict[str, Any] | None
    completeness: dict[str, Any]
    needs_confirmation: list[dict[str, Any]]
    stages: dict[str, Any]


class WorkflowRunRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    query: str = ""
    profile: UserProfile | None = None
    transactions: list[dict[str, Any]] | None = None


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=5, ge=1, le=50)
    memory_types: list[str] | None = None


class AskRequest(BaseModel):
    """A question answered from retrieved profile + memory (the RAG read path)."""

    user_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2_000)
    profile: UserProfile | None = None


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    providers: dict[str, bool]
    embedding_backend: str
    workflows: int
    agents: int
