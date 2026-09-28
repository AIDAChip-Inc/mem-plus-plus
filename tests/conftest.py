"""Shared fixtures + markers for the E2E / behavior suite (``tests/``).

This suite is Burhan's PROOF layer — distinct from ``tests_engine/`` (Awsi's unit
tests, which patch a deterministic FAKE embedder). Here the philosophy is:

* **Always-runnable** tests need no DB, no network, no model — pure imports, the
  SQLite-degraded public-API smoke, the metric known-answers, the migration
  single-head. They run everywhere, including CI with nothing provisioned.
* **Embedder-gated** tests (``@pytest.mark.embedder``) exercise the REAL ONNX
  MiniLM-384 model in-process. They need ``onnxruntime`` + ``tokenizers`` and,
  on a cold cache, network to HuggingFace — but NO database. They SKIP with a
  reason when the model cannot load.
* **DB-gated** tests (``@pytest.mark.db``) exercise the real Postgres 16 +
  pgvector path — the part that cannot run without a container runtime. They
  SKIP with a reason when ``MEMORY_DATABASE_URL`` is unset/unreachable.

Gated tests SKIP (never pass vacuously) when their dependency is absent. A gated
test that finds its dependency PRESENT but the behavior wrong FAILS loudly — in
particular the semantic-leg proof FAILS (not skips) if the vector leg is silently
absent when Postgres is up.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from memory.embeddings import get_embedding_service, reset_registry
from memory.models import Base

FIXTURES = Path(__file__).parent / "fixtures"
TINY_LOCOMO = FIXTURES / "tiny_locomo.json"


def pytest_configure(config: pytest.Config) -> None:
    """Register the gating markers here (pyproject is owned by another persona
    this wave), so ``-m embedder`` / ``-m db`` select cleanly and unknown-marker
    warnings do not fire."""
    config.addinivalue_line(
        "markers", "embedder: needs the real in-process ONNX MiniLM-384 embedder (no DB)"
    )
    config.addinivalue_line(
        "markers", "db: needs a reachable Postgres 16 + pgvector (MEMORY_DATABASE_URL)"
    )


# ── deterministic scope ─────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _fixed_memory_user(monkeypatch):
    """Pin the contributor slug so scope (user_id + by:<slug>) is deterministic."""
    monkeypatch.setenv("MEMORY_USER", "burhan-e2e")


# ── always-runnable: SQLite-degraded engine bound to the public API ──────────────
@pytest.fixture()
def sqlite_public_api(monkeypatch):
    """Bind ``memory.db`` to a fresh in-memory SQLite so ``recall.*`` public
    functions run end-to-end on the DEGRADED path (no embedding, no tsv, LIKE
    lexical + tag legs, vector leg structurally absent). Yields the engine."""
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    yield engine
    engine.dispose()


# ── embedder-gated: the REAL ONNX MiniLM-384 model, no DB ────────────────────────
@pytest.fixture(scope="session")
def real_embedder():
    """The real in-process ONNX MiniLM-384 provider, warm-loaded. SKIPS when the
    model cannot load (onnxruntime/tokenizers missing, or cold cache with no
    network)."""
    reset_registry()
    svc = get_embedding_service("memory")
    if not svc.warm_load():
        pytest.skip(
            "real ONNX MiniLM-384 embedder could not load (needs onnxruntime + "
            "tokenizers, and network to HuggingFace on a cold cache) — "
            "embedder-gated tests skipped"
        )
    return svc


# ── DB-gated: real Postgres 16 + pgvector ────────────────────────────────────────
@pytest.fixture(scope="session")
def pg_engine():
    """Engine over MEMORY_DATABASE_URL with a clean schema. SKIPS when the DSN is
    unset or the server is unreachable, so the suite still runs offline."""
    dsn = os.environ.get("MEMORY_DATABASE_URL")
    if not dsn:
        pytest.skip(
            "MEMORY_DATABASE_URL not set — Postgres+pgvector DB-gated tests "
            "skipped. Start it: `docker compose up -d` then export the DSN "
            "from .env.example."
        )
    if dsn.startswith("sqlite"):
        pytest.skip("MEMORY_DATABASE_URL is SQLite — DB-gated tests need Postgres+pgvector")
    engine = create_engine(dsn, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:  # unreachable server
        pytest.skip(f"Postgres unreachable ({exc}) — DB-gated tests skipped")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def pg_public_api(pg_engine, monkeypatch):
    """Bind ``memory.db`` to the real Postgres engine so ``recall.*`` public
    functions run the FULL path (real embedding write + pgvector recall). The
    REAL embedder is used deliberately — NOT patched to a fake — because these
    tests exist to prove the ONNX+pgvector semantic leg. Yields the engine."""
    Session = sessionmaker(bind=pg_engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", pg_engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    # Fresh scope isolation per test: wipe rows so recall assertions are clean.
    with Session() as s:
        for tbl in ("memory_tag_link", "memory_tag", "agent_memory"):
            s.execute(text(f"DELETE FROM {tbl}"))
        s.commit()
    return pg_engine


def new_uuids():
    """Fresh (user, project, customer) scope tuple for a store-level test."""
    return uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
