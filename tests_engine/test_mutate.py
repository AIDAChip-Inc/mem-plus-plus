"""Unit tests for the manual mutation ops (``memory.mutate``).

Two tiers:

* **SQLite-degraded** (always-runnable, no DB/network/model): proves the
  bi-temporal / consolidation-forest bookkeeping — supersede is soft (no hard
  delete) and drops the row from recall; merge builds a parent, links children via
  ``parent_id``, and supersedes them; reset empties the tables. Embedding is a
  structural no-op on SQLite, so those asserts live in the PG tier.
* **Postgres+pgvector** (``MEMORY_DATABASE_URL`` set, else SKIP): proves the merge
  parent is actually re-embedded and recallable via the vector leg.

The mutation ops open their OWN sessions via ``memory.db``; the fixtures rebind
that global engine (and pin ``MEMORY_USER`` for a deterministic scope) exactly as
the shipped suites do.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from memory import mutate, recall
from memory.models import AgentMemory, Base, MemoryTag, MemoryTagLink


# ── SQLite-degraded fixture ──────────────────────────────────────────────────────
@pytest.fixture()
def sqlite_env(monkeypatch):
    """Fresh in-memory SQLite bound to ``memory.db`` + a pinned contributor slug.

    Yields a ``sessionmaker`` for direct row assertions. ``recall.*`` / ``mutate.*``
    open their own sessions against the same rebound engine."""
    monkeypatch.setenv("MEMORY_USER", "awsi-mutate")
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    yield Session
    engine.dispose()


def _seed(persona: str, summary: str, tags=None) -> str:
    """Store one fact via the shipped path and return its id (querying the row)."""
    recall.store_facts_verbatim(persona, [(summary, tags or [], None)])
    from memory.db import get_session

    db = get_session()
    try:
        row = (
            db.query(AgentMemory)
            .filter(AgentMemory.context_summary == summary)
            .order_by(AgentMemory.created_at.desc())
            .first()
        )
        return str(row.id)
    finally:
        db.close()


# ── supersede ────────────────────────────────────────────────────────────────────
def test_supersede_is_soft_and_drops_from_recall(sqlite_env):
    mid = _seed("awsi", "Adopt pgvector for semantic recall", tags=["pgvector"])
    res = mutate.supersede_memory(mid)
    assert res["id"] == mid
    assert res["superseded_by"] is None
    assert res["valid_to"] is not None

    with sqlite_env() as s:
        row = s.get(AgentMemory, uuid.UUID(mid))
        assert row is not None  # NOT hard-deleted — still present, auditable
        assert row.is_active is False
        assert row.valid_to is not None
        assert row.is_current is False

    # recall filters is_active -> the retired fact is no longer returned.
    rows = recall.recall_facts("awsi", "pgvector recall", k=10)
    assert all("pgvector" not in r["summary"].lower() for r in rows)


def test_supersede_sets_superseded_by_pointer(sqlite_env):
    old = _seed("awsi", "Tapeout target is Q2", tags=["tapeout"])
    new = _seed("awsi", "Tapeout target is Q3", tags=["tapeout"])
    res = mutate.supersede_memory(old, superseded_by=new, reason="date slipped")
    assert res["superseded_by"] == new
    with sqlite_env() as s:
        row = s.get(AgentMemory, uuid.UUID(old))
        assert str(row.superseded_by_id) == new
        assert row.is_active is False


def test_supersede_reason_is_accepted_not_persisted(sqlite_env):
    mid = _seed("awsi", "A fact with a supersede reason")
    # reason is a valid kwarg (Zain codes to it) but has no schema column.
    res = mutate.supersede_memory(mid, reason="user retired it")
    assert res["id"] == mid


def test_supersede_bad_id_raises(sqlite_env):
    with pytest.raises(ValueError, match="invalid memory id"):
        mutate.supersede_memory("not-a-uuid")


def test_supersede_missing_id_raises(sqlite_env):
    with pytest.raises(ValueError, match="not found"):
        mutate.supersede_memory(str(uuid.uuid4()))


# ── merge ──────────────────────────────────────────────────────────────────────
def test_merge_builds_parent_and_supersedes_children(sqlite_env):
    a = _seed("awsi", "The recall engine ports store.py", tags=["engine"])
    b = _seed("awsi", "The recall engine uses RRF fusion", tags=["rrf"])
    res = mutate.merge_memories([a, b])

    assert res["merged_count"] == 2
    assert set(res["child_ids"]) == {a, b}
    parent_id = res["parent_id"]

    with sqlite_env() as s:
        parent = s.get(AgentMemory, uuid.UUID(parent_id))
        assert parent is not None
        assert parent.is_active is True
        assert parent.parent_id is None
        assert parent.hit_count == 0
        # union of both children's tags is preserved as provenance.
        assert {t.label for t in parent.tags} >= {"engine", "rrf"}

        for cid in (a, b):
            child = s.get(AgentMemory, uuid.UUID(cid))
            assert child.is_active is False               # archived
            assert str(child.parent_id) == parent_id      # forest edge -> parent
            assert str(child.superseded_by_id) == parent_id
            assert child.valid_to is not None             # bi-temporal audit trail


def test_merge_deterministic_summary_is_union_of_children(sqlite_env):
    a = _seed("awsi", "PLL loop filter reduces phase noise")
    b = _seed("awsi", "PLL locks in under 10 microseconds")
    res = mutate.merge_memories([a, b])
    with sqlite_env() as s:
        parent = s.get(AgentMemory, uuid.UUID(res["parent_id"]))
        # No LLM: both distinct facts survive verbatim in the deterministic union.
        assert "phase noise" in parent.context_summary
        assert "10 microseconds" in parent.context_summary


def test_merge_uses_provided_summary_when_given(sqlite_env):
    a = _seed("awsi", "detail one")
    b = _seed("awsi", "detail two")
    res = mutate.merge_memories([a, b], summary="a curated merged summary")
    with sqlite_env() as s:
        parent = s.get(AgentMemory, uuid.UUID(res["parent_id"]))
        assert parent.context_summary == "a curated merged summary"


def test_merge_requires_two_distinct_ids(sqlite_env):
    a = _seed("awsi", "only one fact")
    with pytest.raises(ValueError, match="at least 2 distinct"):
        mutate.merge_memories([a, a])  # a double-select is not two children


def test_merge_mixed_scope_raises(sqlite_env):
    mine = _seed("awsi", "fact authored by awsi", tags=["x"])
    theirs = _seed("dina", "fact authored by dina", tags=["x"])
    # Same user/project/customer but different agent_type -> mixed scope.
    with pytest.raises(ValueError, match="mixed scopes"):
        mutate.merge_memories([mine, theirs])


# ── reset ────────────────────────────────────────────────────────────────────────
def test_reset_empties_all_three_tables_and_is_idempotent(sqlite_env):
    _seed("awsi", "fact one", tags=["a"])
    _seed("awsi", "fact two", tags=["b"])
    res = mutate.reset_db()
    assert res["deleted_memories"] == 2
    assert res["deleted_tags"] >= 2  # entity tags + the by:<slug> authorship tag

    with sqlite_env() as s:
        assert s.query(AgentMemory).count() == 0
        assert s.query(MemoryTag).count() == 0
        assert s.query(MemoryTagLink).count() == 0

    # Idempotent: a second reset deletes nothing.
    again = mutate.reset_db()
    assert again == {"deleted_memories": 0, "deleted_tags": 0}


# ── Postgres + pgvector tier (embedding-specific; SKIP without a DB) ───────────────
_DSN = os.environ.get("MEMORY_DATABASE_URL")
_pg = pytest.mark.skipif(
    not _DSN or _DSN.startswith("sqlite"),
    reason="MEMORY_DATABASE_URL unset/sqlite — merge-embedding tier needs Postgres+pgvector",
)


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "awsi-mutate-pg")
    engine = create_engine(_DSN, future=True, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    with Session() as s:
        for tbl in ("memory_tag_link", "memory_tag", "agent_memory"):
            s.execute(text(f"DELETE FROM {tbl}"))
        s.commit()
    yield Session
    engine.dispose()


@_pg
def test_merge_parent_is_embedded_and_recallable_on_pg(pg_env):
    a = _seed("awsi", "The PLL loop filter reduces phase noise", tags=["pll"])
    b = _seed("awsi", "The PLL loop filter improves jitter", tags=["pll"])
    res = mutate.merge_memories([a, b])
    with pg_env() as s:
        parent = s.get(AgentMemory, uuid.UUID(res["parent_id"]))
        assert parent.embedding is not None  # re-embedded via store path
        assert parent.content_tsv is not None
    rows = recall.recall_facts("awsi", "PLL loop filter phase noise", k=10)
    assert any(r["summary"] == parent.context_summary for r in rows)
