"""Alembic environment — resolves the DSN from MEMORY_DATABASE_URL (plain local)."""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from memory.models import Base

config = context.config

_dsn = os.environ.get("MEMORY_DATABASE_URL")
if not _dsn:
    raise RuntimeError(
        "MEMORY_DATABASE_URL is not set. Start the local Postgres via "
        "`docker compose up -d` and export the DSN (see .env.example)."
    )
config.set_main_option("sqlalchemy.url", _dsn)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_dsn, target_metadata=target_metadata, literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
