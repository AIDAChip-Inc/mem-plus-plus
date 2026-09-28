"""Shared fixtures for the engine unit tests.

Pure-function tests need nothing here. DB-backed tests use ``db_session``, which
SKIPS when ``MEMORY_DATABASE_URL`` points at an unreachable Postgres (so the suite
still runs offline), and patches the embedder to a deterministic FAKE so tests are
fast and network-free — the real ONNX MiniLM path is exercised by the manual
docker-compose self-verify, not the unit suite.
"""
from __future__ import annotations

import hashlib
import math
import os

import pytest
from sqlalchemy import create_engine, text

from memory import config
from memory.models import Base


class _FakeEmbedder:
    """Deterministic unit vector from a text hash — no model download."""

    dimensions = config.EMBEDDING_DIM

    def embed(self, text_: str, *, intent: str = "document"):
        if not text_ or text_.isspace():
            return None
        digest = hashlib.sha256(text_.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] - 128) / 128.0 for i in range(config.EMBEDDING_DIM)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def warm_load(self) -> bool:
        return True


@pytest.fixture(scope="session")
def _engine():
    dsn = os.environ.get("MEMORY_DATABASE_URL")
    if not dsn:
        pytest.skip("MEMORY_DATABASE_URL not set — DB-backed tests skipped")
    eng = create_engine(dsn, future=True)
    try:
        with eng.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:  # unreachable Postgres
        pytest.skip(f"Postgres unreachable ({exc}) — DB-backed tests skipped")
    # Clean slate for the unit suite (dedicated docker DB).
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def db_session(_engine, monkeypatch):
    from sqlalchemy.orm import sessionmaker

    # Patch the embedder everywhere the store resolves it.
    fake = _FakeEmbedder()
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _preset: fake)
    # recall_facts / store_facts_verbatim open their own sessions via memory.db —
    # bind that global engine to the test engine so they hit the same schema.
    monkeypatch.setattr("memory.db._engine", _engine, raising=False)
    monkeypatch.setattr(
        "memory.db._SessionLocal",
        sessionmaker(bind=_engine, expire_on_commit=True, future=True),
        raising=False,
    )

    Session = sessionmaker(bind=_engine, expire_on_commit=True, future=True)
    session = Session()
    try:
        yield session
    finally:
        session.close()
