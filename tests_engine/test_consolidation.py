"""Unit tests for the synthesize-merge consolidation (``memory.consolidation``).

Three tiers:

* **Pure** (no DB / no network): the DB-free helpers — ``dedup_groups`` near-dup
  grouping, ``classify_relation`` safe-default mapping, ``deterministic_union_summary``.
* **SQLite-degraded** (always-runnable, offline): the full ``consolidate(persona)``
  entry over hand-set embeddings (SQLite stores the pgvector column as a list, so
  numpy cosine grouping runs exactly as on Postgres) — near-dups merge into a
  parent, children archived, the ``MemoryConsolidationRun`` ledger row is written,
  never-merge-tagged rows are skipped, the CONFLICT path supersedes newest-wins,
  and the M5 two-section write path populates ``agent_summary``.
* **Postgres+pgvector** (``MEMORY_DATABASE_URL`` set, else SKIP): the merge parent
  is actually re-embedded via the store path and recallable.

The consolidation ops open their OWN sessions via ``memory.db``; the fixtures
rebind that global engine (and pin ``MEMORY_USER`` for a deterministic scope)
exactly as the shipped suites do. LLM availability is monkeypatched per-test so the
suite is deterministic and never touches the network.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from memory import consolidation, recall
from memory.consolidation import GroupRelation
from memory.db import get_session
from memory.models import AgentMemory, Base, MemoryConsolidationRun
from memory.store import make_scope_key


# ── pure helpers (no DB) ──────────────────────────────────────────────────────
def _vec(*head: float) -> list[float]:
    """A 384-d vector with the given leading components (rest zero)."""
    v = [0.0] * 384
    for i, x in enumerate(head):
        v[i] = x
    return v


NEAR_A = _vec(1.0, 0.0)
NEAR_B = _vec(0.98, 0.02)   # cosine ~0.9998 with NEAR_A -> grouped at tau=0.85
NEAR_C = _vec(0.95, 0.05)   # also near NEAR_A
FAR = _vec(0.0, 1.0)        # cosine 0 with NEAR_A -> never grouped


def test_dedup_groups_finds_near_dupes_and_excludes_far():
    groups = consolidation.dedup_groups([NEAR_A, NEAR_B, FAR], tau=0.85)
    assert groups == [[0, 1]]  # the two near-dups; FAR is a singleton (dropped)


def test_dedup_groups_disjoint_and_singletons_dropped():
    # Three mutually-near vectors -> one group of 3; a lone FAR -> no group.
    groups = consolidation.dedup_groups([NEAR_A, NEAR_B, NEAR_C, FAR], tau=0.85)
    assert groups == [[0, 1, 2]]


def test_dedup_groups_threshold_gates():
    # At tau=0.999 the looser NEAR_C (cosine ~0.9986 with A) drops out.
    groups = consolidation.dedup_groups([NEAR_A, NEAR_C], tau=0.999)
    assert groups == []


def test_classify_relation_safe_default():
    assert consolidation.classify_relation(None) is GroupRelation.DISTINCT_FACTS
    assert consolidation.classify_relation({"relation": "bogus"}) is GroupRelation.DISTINCT_FACTS
    assert consolidation.classify_relation({"relation": "conflict"}) is GroupRelation.CONFLICT
    assert consolidation.classify_relation({"relation": "restatement"}) is GroupRelation.RESTATEMENT


class _Row:
    def __init__(self, summary):
        self.context_summary = summary
        self.content = summary


def test_deterministic_union_summary_is_lossless_union():
    rows = [_Row("pin count is 32"), _Row("host is db-01"), _Row("pin count is 32")]
    out = consolidation.deterministic_union_summary(rows)
    # distinct details survive; the duplicate is collapsed.
    assert "pin count is 32" in out
    assert "host is db-01" in out
    assert out.count("pin count is 32") == 1


# ── SQLite-degraded fixture ────────────────────────────────────────────────────
@pytest.fixture()
def sqlite_env(monkeypatch):
    """Fresh in-memory SQLite bound to ``memory.db`` + a pinned contributor slug."""
    monkeypatch.setenv("MEMORY_USER", "awsi-consolidation")
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    # Default: force the offline (deterministic-union) path — no network, reproducible.
    monkeypatch.setattr(consolidation, "llm_available", lambda: False)
    yield Session
    engine.dispose()


def _seed(persona, summary, *, tags=None, embedding=None, occurred_at=None):
    """Store one fact via the shipped path, optionally stamp an embedding, return id."""
    recall.store_facts_verbatim(persona, [(summary, tags or [], occurred_at)])
    db = get_session()
    try:
        row = (
            db.query(AgentMemory)
            .filter(AgentMemory.context_summary == summary)
            .order_by(AgentMemory.created_at.desc())
            .first()
        )
        if embedding is not None:
            row.embedding = embedding
            db.commit()
        return str(row.id)
    finally:
        db.close()


def test_consolidate_merges_near_dupes_offline(sqlite_env):
    a = _seed("awsi", "pgvector adopted for recall", tags=["pgvector"], embedding=NEAR_A)
    b = _seed("awsi", "we use pgvector for semantic recall", tags=["semantic"], embedding=NEAR_B)
    far = _seed("awsi", "tapeout target is Q3", tags=["tapeout"], embedding=FAR)

    res = consolidation.consolidate("awsi", threshold=0.85)

    assert res["pools_merged"] == 1
    assert res["superseded"] == 0            # offline path does not classify conflicts
    assert res["memories_before"] == 3
    assert res["memories_after"] == 2        # 2 children archived, 1 parent added, FAR intact
    assert res["run_id"]

    with sqlite_env() as s:
        # the two near-dups are archived under a new active parent
        child_a = s.get(AgentMemory, uuid.UUID(a))
        child_b = s.get(AgentMemory, uuid.UUID(b))
        assert child_a.is_active is False and child_b.is_active is False
        assert child_a.parent_id == child_b.parent_id is not None
        parent = s.get(AgentMemory, child_a.parent_id)
        assert parent.is_active is True and parent.parent_id is None and parent.hit_count == 0
        # deterministic union preserves BOTH children's distinct text
        assert "pgvector adopted for recall" in parent.context_summary
        assert "semantic recall" in parent.context_summary
        # union-of-tags provenance
        assert {t.label for t in parent.tags} >= {"pgvector", "semantic"}
        # the FAR fact is untouched
        assert s.get(AgentMemory, uuid.UUID(far)).is_active is True


def test_consolidate_writes_ledger_row(sqlite_env):
    _seed("awsi", "fact one about the bus", embedding=NEAR_A)
    _seed("awsi", "fact one re the message bus", embedding=NEAR_B)

    res = consolidation.consolidate("awsi", threshold=0.85)

    uid, pid, _cid, _slug = recall._scope()
    expected_scope = make_scope_key(uid, "awsi", pid)
    with sqlite_env() as s:
        rows = s.query(MemoryConsolidationRun).all()
        assert len(rows) == 1
        ledger = rows[0]
        assert ledger.scope_key == expected_scope
        assert str(ledger.id) == res["run_id"]
        assert ledger.groups_merged == res["pools_merged"] == 1
        assert ledger.rows_archived == 2
        assert ledger.last_run_at is not None


def test_consolidate_run_is_idempotent_upsert(sqlite_env):
    _seed("awsi", "alpha the pll locks fast", embedding=NEAR_A)
    _seed("awsi", "alpha pll locks quickly", embedding=NEAR_B)
    first = consolidation.consolidate("awsi", threshold=0.85)
    # A second run: nothing new to merge (archived sources dropped out) -> upsert,
    # NOT a duplicate ledger row (scope_key is unique).
    second = consolidation.consolidate("awsi", threshold=0.85)
    assert second["pools_merged"] == 0
    assert second["run_id"] == first["run_id"]
    with sqlite_env() as s:
        assert s.query(MemoryConsolidationRun).count() == 1


def test_consolidate_skips_never_merge_tagged(sqlite_env):
    a = _seed("awsi", "the clock tree uses balanced buffers", embedding=NEAR_A)
    pinned = _seed("awsi", "clock tree balanced buffers pinned", tags=["pinned"], embedding=NEAR_B)
    c = _seed("awsi", "clock tree relies on balanced buffers", embedding=NEAR_C)

    res = consolidation.consolidate("awsi", threshold=0.85)

    assert res["pools_merged"] == 1  # a + c merged; pinned excluded from the pool
    with sqlite_env() as s:
        pinned_row = s.get(AgentMemory, uuid.UUID(pinned))
        assert pinned_row.is_active is True      # never archived
        assert pinned_row.parent_id is None      # never merged into a parent
        assert s.get(AgentMemory, uuid.UUID(a)).is_active is False
        assert s.get(AgentMemory, uuid.UUID(c)).is_active is False


def test_consolidate_supersedes_conflict_newest_wins(sqlite_env, monkeypatch):
    # Force the LLM path ON and classify the group as a value-CONFLICT.
    monkeypatch.setattr(consolidation, "llm_available", lambda: True)
    monkeypatch.setattr(consolidation, "classify_group", lambda texts: GroupRelation.CONFLICT)

    old = _seed(
        "awsi", "the pin count is 32", embedding=NEAR_A,
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    new = _seed(
        "awsi", "the pin count is now 40", embedding=NEAR_B,
        occurred_at=datetime(2024, 6, 1, tzinfo=UTC),
    )

    res = consolidation.consolidate("awsi", threshold=0.85)

    assert res["superseded"] == 1
    assert res["pools_merged"] == 0
    # Conflict-supersede keeps is_active=True (queryable-A) -> active count unchanged.
    assert res["memories_after"] == res["memories_before"] == 2
    with sqlite_env() as s:
        old_row = s.get(AgentMemory, uuid.UUID(old))
        new_row = s.get(AgentMemory, uuid.UUID(new))
        assert old_row.valid_to is not None            # older fact retired-in-time
        assert str(old_row.superseded_by_id) == new     # points at the winner
        assert old_row.is_active is True                # STILL recallable (differs from manual retire)
        assert new_row.valid_to is None                 # newest stays current


def test_agent_summary_populated_on_two_section_write(sqlite_env, monkeypatch):
    # M5: the gated two-section write path captures agent_summary. Stub the LLM
    # extractor (offline) to return a fixed two-section tuple.
    monkeypatch.setattr(
        "memory.recall.extract_two_section",
        lambda content, **kw: ("pgvector chosen for recall", ["pgvector"], None,
                               "the agent confirmed pgvector is the choice"),
    )
    out = recall.store_facts("awsi", "user turn about vector search", mode="gated")
    assert out["written"] == 1
    with sqlite_env() as s:
        row = (
            s.query(AgentMemory)
            .filter(AgentMemory.context_summary == "pgvector chosen for recall")
            .first()
        )
        assert row.agent_summary == "the agent confirmed pgvector is the choice"


def test_verbatim_write_leaves_agent_summary_null(sqlite_env):
    _seed("awsi", "a verbatim stored fact with no agent section")
    with sqlite_env() as s:
        row = (
            s.query(AgentMemory)
            .filter(AgentMemory.context_summary == "a verbatim stored fact with no agent section")
            .first()
        )
        assert row.agent_summary is None


# ── Postgres + pgvector tier (embedding re-embed; SKIP without a DB) ────────────
_DSN = os.environ.get("MEMORY_DATABASE_URL")
_pg = pytest.mark.skipif(
    not _DSN or _DSN.startswith("sqlite"),
    reason="MEMORY_DATABASE_URL unset/sqlite — consolidation re-embed tier needs Postgres+pgvector",
)


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "awsi-consolidation-pg")
    engine = create_engine(_DSN, future=True, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    monkeypatch.setattr(consolidation, "llm_available", lambda: False)
    with Session() as s:
        for tbl in ("memory_consolidation_run", "memory_tag_link", "memory_tag", "agent_memory"):
            s.execute(text(f"DELETE FROM {tbl}"))
        s.commit()
    yield Session
    engine.dispose()


@_pg
def test_consolidate_parent_is_re_embedded_on_pg(pg_env):
    # Hand-set near-dup child embeddings (the fake/real embedder can't force text
    # paraphrases to be near-dups); assert the PARENT is re-embedded via the store path.
    _seed("awsi", "the bus uses Kafka for durability", embedding=NEAR_A)
    _seed("awsi", "durability on the bus comes from Kafka", embedding=NEAR_B)
    res = consolidation.consolidate("awsi", threshold=0.85)
    assert res["pools_merged"] == 1
    with pg_env() as s:
        parent = s.query(AgentMemory).filter(
            AgentMemory.is_active.is_(True), AgentMemory.parent_id.is_(None)
        ).first()
        assert parent.embedding is not None    # re-embedded via _populate_search_fields
        assert parent.content_tsv is not None
