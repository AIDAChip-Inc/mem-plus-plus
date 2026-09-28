"""Agent working-memory ORM — faithful replica of production ``app/models/memory.py``.

Three tables:
- ``agent_memory``    — one row per memory
- ``memory_tag``      — per-scope canonical tag registry, UNIQUE(scope_key, label)
- ``memory_tag_link`` — memory <-> tag many-to-many (composite PK)

Faithful to production EXCEPT the external foreign keys. Production points
``user_id``/``customer_id``/``project_id``/``session_id``/``source_message_id`` at
``users``/``customers``/``projects``/``chat_sessions``/``chat_messages``. This
self-contained replica has no such tables, so those become PLAIN GUID columns (no
FK). KEPT verbatim: the pgvector ``Vector(384)`` embedding + its HNSW index, the
``content_tsv`` TSVECTOR + GIN index, the composite ``ix_mem_scope`` recency index,
and the self-referential FKs (``parent_id``, ``superseded_by_id``).

``content_tsv`` is computed in the SERVICE layer at insert time via
``func.to_tsvector('english', ...)`` on PostgreSQL — no DB trigger (see
``store.PostgresMemoryStore.write``). On SQLite it degrades to plain Text.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for the memory-research schema."""


class GUID(TypeDecorator):
    """Platform-independent GUID — PG UUID, else CHAR(36). Ported verbatim."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        return uuid.UUID(value)


class FlexibleTSVector(TypeDecorator):
    """TSVECTOR on PostgreSQL, Text on SQLite (no-op for search). Ported verbatim."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TSVECTOR())
        return dialect.type_descriptor(Text())


class MemoryTier(enum.StrEnum):
    """Memory tiers. This replica uses only AGENT_WORKING."""

    AGENT_WORKING = "agent_working"
    PROJECT = "project"
    TRIBAL_KB = "tribal_kb"


class AgentMemory(Base):
    """One agent memory. Scope coordinates: (user_id, agent_type, project_id).

    ``session_id`` is provenance only — recall deliberately omits it so
    cross-session recall works.
    """

    __tablename__ = "agent_memory"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    occurred_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Hit tracking (CPU-cache analogy: count + which agent hit it)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_hit_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    memory_tier: Mapped[str] = mapped_column(
        String(32), default=MemoryTier.AGENT_WORKING.value, nullable=False
    )

    # Scope coordinates — PLAIN GUID columns (no external FKs in this replica).
    user_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    customer_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)  # provenance only
    agent_type: Mapped[str] = mapped_column(String(64), nullable=False)
    discipline: Mapped[str | None] = mapped_column(String(64), nullable=True)
    authority: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Bi-temporal validity for queryable supersession. A fact is CURRENT iff
    # valid_to IS NULL. valid_from backfills to occurred_at else created_at.
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Self-FK provenance: superseded -> current (SET NULL, never cascade).
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("agent_memory.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Content + summary + full-text vector (PG only; service-layer on insert).
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Two-section cell agent-stated section: recall PAYLOAD only — never embedded,
    # never in content_tsv.
    agent_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_tsv: Mapped[str | None] = mapped_column(FlexibleTSVector(), nullable=True)
    # Semantic embedding (MiniLM-384, PG-only). NULL -> recall's vector leg skipped.
    embedding = mapped_column(Vector(384), nullable=True)

    # Rank attribute — refreshed at recall time (query-dependent).
    salience: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Idempotency key — the source message id (plain GUID, no FK here).
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)

    # Consolidation forest edge (self-FK). SET NULL — archive, never hard-delete.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("agent_memory.id", ondelete="SET NULL"), nullable=True, index=True
    )

    tags: Mapped[list["MemoryTag"]] = relationship(
        "MemoryTag", secondary="memory_tag_link", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_mem_tsv", "content_tsv", postgresql_using="gin"),
        Index(
            "ix_mem_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    @property
    def is_current(self) -> bool:
        """A fact is CURRENT iff it has no ``valid_to``."""
        return self.valid_to is None

    def __repr__(self) -> str:
        return (
            f"<AgentMemory(id={str(self.id)[:8]}, agent={self.agent_type}, "
            f"hits={self.hit_count}, active={self.is_active})>"
        )


# Composite scope index with DESC recency — declared post-class so the DESC
# ordering can use the column attribute (Appendix A: ix_mem_scope).
Index(
    "ix_mem_scope",
    AgentMemory.user_id,
    AgentMemory.agent_type,
    AgentMemory.project_id,
    AgentMemory.is_active,
    AgentMemory.created_at.desc(),
)


class MemoryTag(Base):
    """Canonical tag registry, per scope.

    ``scope_key`` = ``f"{user_id}|{agent_type}|{project_id or 'global'}"``.
    """

    __tablename__ = "memory_tag"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (UniqueConstraint("scope_key", "label"),)

    def __repr__(self) -> str:
        return f"<MemoryTag(label={self.label}, scope={self.scope_key})>"


class MemoryTagLink(Base):
    """Memory <-> tag many-to-many link (composite PK)."""

    __tablename__ = "memory_tag_link"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("agent_memory.id"), primary_key=True
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("memory_tag.id"), primary_key=True, index=True
    )


class MemoryConsolidationRun(Base):
    """Durable ledger for a synthesize-merge consolidation pass — faithful replica
    of production ``app/models/memory.py`` ``MemoryConsolidationRun``.

    One upserted row per consolidation ``scope_key`` (``user|agent|project``).
    ``last_run_at`` is the run timing; the counters are the audit trail
    ("Agent IS the Dashboard"). Columns mirror production EXACTLY — production has
    no threshold/duration column, so ``consolidate()``'s threshold is logged, not
    stored. No FKs (``scope_key`` is a string, not a row pointer).
    """

    __tablename__ = "memory_consolidation_run"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    last_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    groups_merged: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_archived: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<MemoryConsolidationRun(scope={self.scope_key}, "
            f"last_run_at={self.last_run_at}, merged={self.groups_merged})>"
        )
