"""
SQLAlchemy models for the memory layer (Neon Postgres + pgvector).

The deck names six memory types. They share one table with a `memory_type`
discriminator rather than six near-identical tables, because every type has the
same shape -- who, what, when, how important, and an embedding -- and semantic
recall must be able to search *across* types in one query.

Tables
------
memories        every remembered episode/fact/pattern/goal/simulation, embedded
user_profiles   the Profile Agent's durable profile document (Day 5)
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..common import config


class Base(DeclarativeBase):
    pass


#: The six memory types from the deck's Memory Layer.
MEMORY_TYPES = (
    "episodic",     # what happened: interactions, decisions, events
    "semantic",     # what is true: durable facts learned about the user
    "behavioral",   # how they act: detected habits, biases, tendencies
    "goal",         # what they want: objectives and progress
    "simulation",   # what was projected: forecast/Monte-Carlo runs, for hindsight
    "community",    # what people like them do: cohort-level patterns
)


def _uuid() -> str:
    return str(uuid.uuid4())


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)

    #: Human-readable text. This is what gets embedded.
    content: Mapped[str] = mapped_column(Text, nullable=False)

    #: Structured payload -- typically an agent's `<agent>_result` dict.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    #: Which agent/workflow wrote this, for provenance and targeted recall.
    source_agent: Mapped[str | None] = mapped_column(String(64), index=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(config.EMBEDDING_DIM))

    #: 0..1. Drives ranking and, later, forgetting.
    importance: Mapped[float] = mapped_column(Float, default=0.5)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    accessed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        # The hot path is "this user's memories of this type, newest first".
        Index("ix_memories_user_type_created", "user_id", "memory_type", "created_at"),
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "payload": self.payload or {},
            "source_agent": self.source_agent,
            "importance": self.importance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "access_count": self.access_count,
        }


class UserProfileRecord(Base):
    """Durable profile document maintained by the Profile Agent."""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
