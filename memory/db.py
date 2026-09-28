"""Engine + session factory — a PLAIN local DSN from the environment.

Production resolves its DSN through Railway/remote/1Password with self-heal,
pool-reset, and retry. All of that is stripped here (this is a local research
replica): the connection is a single ``MEMORY_DATABASE_URL`` env var. The default
path is the Docker-free, project-local Postgres cluster that ``run_demo.sh``
creates in ``./.pgdata`` and serves over a unix socket in ``./.pgsock`` (it
computes + exports the socket DSN for you); docker-compose is an alternative.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _dsn() -> str:
    dsn = os.environ.get("MEMORY_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "MEMORY_DATABASE_URL is not set. Run `./run_demo.sh` (it starts the "
            "project-local Postgres and exports the socket DSN), or set it manually "
            "per .env.example."
        )
    return dsn


def get_engine() -> Engine:
    """Process-wide lazy engine over the local DSN (no pool-reset machinery)."""
    global _engine
    if _engine is None:
        _engine = create_engine(_dsn(), future=True, pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    """A new ORM session bound to the local engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=True, future=True)
    return _SessionLocal()
