"""initial memory-research schema (agent_memory + memory_tag + memory_tag_link)

Creates the pgvector extension, the three tables, and the KEPT indexes:
the composite ``ix_mem_scope`` recency index, the ``content_tsv`` GIN index, and
the MiniLM-384 embedding HNSW (cosine) index.

Revision ID: 0001
Revises:
Create Date: 2026-07-23
"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_GUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "agent_memory",
        sa.Column("id", _GUID, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hit_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_hit_by", sa.String(length=64), nullable=True),
        sa.Column("memory_tier", sa.String(length=32), server_default="agent_working", nullable=False),
        # Scope coordinates — plain GUID columns (no external FKs in this replica).
        sa.Column("user_id", _GUID, nullable=False),
        sa.Column("customer_id", _GUID, nullable=True),
        sa.Column("project_id", _GUID, nullable=True),
        sa.Column("session_id", _GUID, nullable=True),
        sa.Column("agent_type", sa.String(length=64), nullable=False),
        sa.Column("discipline", sa.String(length=64), nullable=True),
        sa.Column("authority", sa.String(length=64), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by_id", _GUID, nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_summary", sa.Text(), nullable=True),
        sa.Column("agent_summary", sa.Text(), nullable=True),
        sa.Column("content_tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column("salience", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("source_message_id", _GUID, nullable=True),
        sa.Column("parent_id", _GUID, nullable=True),
        # Self-referential FKs (KEPT): SET NULL — archive, never hard-delete.
        sa.ForeignKeyConstraint(["superseded_by_id"], ["agent_memory.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["parent_id"], ["agent_memory.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_agent_memory_occurred_at", "agent_memory", ["occurred_at"])
    op.create_index("ix_agent_memory_superseded_by_id", "agent_memory", ["superseded_by_id"])
    op.create_index("ix_agent_memory_source_message_id", "agent_memory", ["source_message_id"])
    op.create_index("ix_agent_memory_parent_id", "agent_memory", ["parent_id"])
    # Composite scope index with DESC recency (Appendix A: ix_mem_scope).
    op.create_index(
        "ix_mem_scope",
        "agent_memory",
        ["user_id", "agent_type", "project_id", "is_active", sa.text("created_at DESC")],
    )
    # Full-text GIN index over content_tsv.
    op.create_index("ix_mem_tsv", "agent_memory", ["content_tsv"], postgresql_using="gin")
    # MiniLM-384 embedding HNSW (cosine) index.
    op.create_index(
        "ix_mem_embedding_hnsw",
        "agent_memory",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "memory_tag",
        sa.Column("id", _GUID, primary_key=True),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.UniqueConstraint("scope_key", "label", name="uq_memory_tag_scope_key"),
    )

    op.create_table(
        "memory_tag_link",
        sa.Column("memory_id", _GUID, primary_key=True),
        sa.Column("tag_id", _GUID, primary_key=True),
        sa.ForeignKeyConstraint(["memory_id"], ["agent_memory.id"]),
        sa.ForeignKeyConstraint(["tag_id"], ["memory_tag.id"]),
    )
    op.create_index("ix_memory_tag_link_tag_id", "memory_tag_link", ["tag_id"])


def downgrade() -> None:
    op.drop_table("memory_tag_link")
    op.drop_table("memory_tag")
    op.drop_index("ix_mem_embedding_hnsw", table_name="agent_memory")
    op.drop_index("ix_mem_tsv", table_name="agent_memory")
    op.drop_index("ix_mem_scope", table_name="agent_memory")
    op.drop_index("ix_agent_memory_parent_id", table_name="agent_memory")
    op.drop_index("ix_agent_memory_source_message_id", table_name="agent_memory")
    op.drop_index("ix_agent_memory_superseded_by_id", table_name="agent_memory")
    op.drop_index("ix_agent_memory_occurred_at", table_name="agent_memory")
    op.drop_table("agent_memory")
