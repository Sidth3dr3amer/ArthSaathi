"""
MemoryStore -- the single interface every agent uses to remember and recall.

Every workflow in the deck ends in "Update Memory"; this is that step.

Two implementations:

  PostgresMemoryStore  Neon + pgvector. Production and demo.
  InMemoryStore        Same semantics, no database. Used by the default test
                       suite so the offline run stays fast and hermetic.

Both satisfy the same contract, so a test that passes against one must pass
against the other -- `tests/memory/test_store.py` runs the shared suite twice.
"""

from __future__ import annotations

import datetime as dt
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from ..common import config
from . import embeddings
from .models import MEMORY_TYPES, Base, Memory, UserProfileRecord


def _validate_type(memory_type: str) -> str:
    if memory_type not in MEMORY_TYPES:
        raise ValueError(
            f"unknown memory_type {memory_type!r}; expected one of {list(MEMORY_TYPES)}"
        )
    return memory_type


class MemoryStore(ABC):
    """What agents may do with memory."""

    @abstractmethod
    def remember(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        payload: dict[str, Any] | None = None,
        source_agent: str | None = None,
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Store one memory and return it."""

    @abstractmethod
    def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        memory_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over a user's memories, most similar first."""

    @abstractmethod
    def recent(
        self,
        user_id: str,
        limit: int = 10,
        memory_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Most recently created memories, newest first."""

    @abstractmethod
    def forget(self, user_id: str, memory_type: str | None = None) -> int:
        """Delete a user's memories. Returns how many were removed."""

    @abstractmethod
    def save_profile(self, user_id: str, profile: dict[str, Any]) -> None: ...

    @abstractmethod
    def load_profile(self, user_id: str) -> dict[str, Any] | None: ...


# --------------------------------------------------------------------------- #
# In-memory implementation (tests, offline demos)
# --------------------------------------------------------------------------- #

class InMemoryStore(MemoryStore):
    """Dict-backed store with identical semantics to the Postgres one."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._profiles: dict[str, dict[str, Any]] = {}
        # Wall-clock timestamps tie when several memories are written in the
        # same instant, which makes "newest first" ambiguous. A monotonic
        # sequence keeps recency deterministic.
        self._seq = 0

    def remember(self, user_id, memory_type, content, payload=None,
                 source_agent=None, importance=0.5):
        _validate_type(memory_type)
        row = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "payload": payload or {},
            "source_agent": source_agent,
            "importance": importance,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "access_count": 0,
            "_seq": self._seq,
            "_embedding": embeddings.embed(content),
        }
        self._seq += 1
        self._rows.append(row)
        return {k: v for k, v in row.items() if not k.startswith("_")}

    def _filter(self, user_id, memory_types):
        rows = [r for r in self._rows if r["user_id"] == user_id]
        if memory_types:
            wanted = {_validate_type(t) for t in memory_types}
            rows = [r for r in rows if r["memory_type"] in wanted]
        return rows

    def recall(self, user_id, query, limit=5, memory_types=None):
        rows = self._filter(user_id, memory_types)
        if not rows:
            return []
        q = embeddings.embed(query)
        scored = [
            (embeddings.cosine_similarity(q, r["_embedding"]), r) for r in rows
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, row in scored[:limit]:
            row["access_count"] += 1
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            public["similarity"] = round(float(score), 6)
            out.append(public)
        return out

    def recent(self, user_id, limit=10, memory_types=None):
        rows = self._filter(user_id, memory_types)
        rows = sorted(rows, key=lambda r: r["_seq"], reverse=True)[:limit]
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    def forget(self, user_id, memory_type=None):
        if memory_type:
            _validate_type(memory_type)
        before = len(self._rows)
        self._rows = [
            r for r in self._rows
            if not (r["user_id"] == user_id
                    and (memory_type is None or r["memory_type"] == memory_type))
        ]
        return before - len(self._rows)

    def save_profile(self, user_id, profile):
        self._profiles[user_id] = dict(profile)

    def load_profile(self, user_id):
        stored = self._profiles.get(user_id)
        return dict(stored) if stored is not None else None


# --------------------------------------------------------------------------- #
# Postgres implementation (Neon + pgvector)
# --------------------------------------------------------------------------- #

def _sqlalchemy_url(url: str) -> str:
    """
    Neon hands out a libpq URL; SQLAlchemy needs an explicit driver.

    `channel_binding` is a libpq connection parameter that psycopg accepts but
    SQLAlchemy's URL parser does not recognise, so it is left in place -- psycopg
    forwards unknown query params to libpq unchanged.
    """
    if url.startswith("postgresql+"):
        return url
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


class PostgresMemoryStore(MemoryStore):
    def __init__(self, url: str | None = None, echo: bool = False) -> None:
        url = url or config.DATABASE_URL
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add your Neon connection string to .env."
            )
        self.engine = create_engine(
            _sqlalchemy_url(url), echo=echo, pool_pre_ping=True
        )
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    # -- schema -----------------------------------------------------------
    def create_schema(self) -> None:
        """Enable pgvector and create the tables. Idempotent."""
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    # -- writes -----------------------------------------------------------
    def remember(self, user_id, memory_type, content, payload=None,
                 source_agent=None, importance=0.5):
        _validate_type(memory_type)
        record = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            payload=payload or {},
            source_agent=source_agent,
            importance=importance,
            embedding=embeddings.embed(content),
        )
        with self._session.begin() as session:
            session.add(record)
        return record.as_dict()

    # -- reads ------------------------------------------------------------
    def recall(self, user_id, query, limit=5, memory_types=None):
        vector = embeddings.embed(query)
        stmt = select(Memory).where(Memory.user_id == user_id)
        if memory_types:
            wanted = [_validate_type(t) for t in memory_types]
            stmt = stmt.where(Memory.memory_type.in_(wanted))
        # cosine distance; 1 - distance == cosine similarity for unit vectors
        stmt = stmt.order_by(Memory.embedding.cosine_distance(vector)).limit(limit)

        with self._session() as session:
            rows = list(session.scalars(stmt))
            out = []
            for row in rows:
                similarity = embeddings.cosine_similarity(vector, list(row.embedding))
                row.access_count = (row.access_count or 0) + 1
                row.accessed_at = dt.datetime.now(dt.timezone.utc)
                payload = row.as_dict()
                payload["similarity"] = round(float(similarity), 6)
                out.append(payload)
            session.commit()
        return out

    def recent(self, user_id, limit=10, memory_types=None):
        stmt = select(Memory).where(Memory.user_id == user_id)
        if memory_types:
            wanted = [_validate_type(t) for t in memory_types]
            stmt = stmt.where(Memory.memory_type.in_(wanted))
        stmt = stmt.order_by(Memory.created_at.desc()).limit(limit)
        with self._session() as session:
            return [r.as_dict() for r in session.scalars(stmt)]

    def forget(self, user_id, memory_type=None):
        if memory_type:
            _validate_type(memory_type)
        stmt = delete(Memory).where(Memory.user_id == user_id)
        if memory_type:
            stmt = stmt.where(Memory.memory_type == memory_type)
        with self._session.begin() as session:
            return session.execute(stmt).rowcount or 0

    # -- profile ----------------------------------------------------------
    def save_profile(self, user_id, profile):
        with self._session.begin() as session:
            existing = session.get(UserProfileRecord, user_id)
            if existing is None:
                session.add(UserProfileRecord(user_id=user_id, profile=dict(profile)))
            else:
                existing.profile = dict(profile)

    def load_profile(self, user_id):
        with self._session() as session:
            record = session.get(UserProfileRecord, user_id)
            return dict(record.profile) if record else None


# --------------------------------------------------------------------------- #
# Default store selection
# --------------------------------------------------------------------------- #

_default: MemoryStore | None = None


def get_store(force_memory: bool = False) -> MemoryStore:
    """
    The process-wide store.

    Uses Postgres when DATABASE_URL is set, otherwise falls back to the
    in-memory implementation so nothing hard-fails without a database.
    """
    global _default
    if _default is None or force_memory:
        if force_memory or not config.DATABASE_URL:
            _default = InMemoryStore()
        else:
            _default = PostgresMemoryStore()
    return _default


def set_store(store: MemoryStore) -> None:
    """Override the process-wide store (used by tests and the API layer)."""
    global _default
    _default = store
