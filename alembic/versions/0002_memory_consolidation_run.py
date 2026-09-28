"""memory_consolidation_run ledger (synthesize-merge consolidation)

Adds the durable per-scope consolidation-run ledger — one upserted row per
consolidation ``scope_key`` recording the run timing (``last_run_at``) and audit
counters (``groups_merged`` / ``rows_archived`` / ``cost_calls``). Faithful to the
production ``memory_consolidation_run`` table. Alembic owns this schema in the
replica (production's ``_ensure_consolidation_schema`` self-heal DDL is dropped).

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-23
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_GUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "memory_consolidation_run",
        sa.Column("id", _GUID, primary_key=True),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("groups_merged", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("rows_archived", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cost_calls", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.UniqueConstraint("scope_key", name="uq_memory_consolidation_run_scope_key"),
    )


def downgrade() -> None:
    op.drop_table("memory_consolidation_run")
