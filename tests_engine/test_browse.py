"""Unit tests for the read-only browse API (``memory.browse``).

Structured filters, lexical text, ordering, pagination and the READ-ONLY
(no-hit-bump) invariant run on the always-available SQLite-degraded path. The
semantic (pgvector cosine) leg is DB-gated: it SKIPS cleanly without a reachable
Postgres, and — when Postgres IS present — FAILS loudly if the vector leg is
silently absent, rather than passing vacuously.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memory import config, recall
from memory.browse import browse_memories, get_memory
from memory.models import AgentMemory, Base

# The faithful full production cell — every ``AgentMemory`` column, JSON-safe.
# ``browse_memories`` rows and ``get_memory`` MUST both expose exactly this set.
_FULL_CELL_KEYS = frozenset({
    "id", "summary", "content", "context_summary", "agent_summary",
    "tags", "by",
    "user_id", "agent_type", "project_id", "customer_id", "session_id", "scope",
    "memory_tier", "discipline", "authority",
    "occurred_at", "created_at", "last_used_at", "valid_from", "valid_to",
    "is_active", "superseded_by_id", "parent_id",
    "hit_count", "last_hit_by", "salience",
    "source_message_id",
    "has_embedding", "embedding_dim", "has_tsv",
})


class _FakeEmbedder:
    """Deterministic unit vector from a text hash — no model download (mirrors the
    ``tests_engine`` PG conftest fake so query and doc vectors share one space)."""

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


@pytest.fixture()
def sqlite_api(monkeypatch):
    """Bind ``memory.db`` to a fresh in-memory SQLite so the public API (write +
    browse) runs end-to-end on the DEGRADED path (no embedding, LIKE lexical)."""
    monkeypatch.setenv("MEMORY_USER", "abdu")
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    yield engine
    engine.dispose()


def _seed(sqlite_api):
    """Two personas, three dated/tagged facts, in the current scope."""
    recall.store_facts_verbatim(
        "awsi",
        [
            ("The PLL loop filter reduces phase noise", ["pll", "phase-noise"],
             datetime(2026, 7, 20, tzinfo=UTC)),
            ("Adopt pgvector for semantic recall", ["pgvector"],
             datetime(2026, 7, 22, tzinfo=UTC)),
        ],
    )
    recall.store_facts_verbatim(
        "dina",
        [("Dina designed the recall engine schema", ["engine", "schema"],
          datetime(2026, 7, 21, tzinfo=UTC))],
    )


# ── structured filters ──────────────────────────────────────────────────────────
def test_persona_filter(sqlite_api):
    _seed(sqlite_api)
    rows = browse_memories(persona="awsi")
    assert rows and all(r["agent_type"] == "awsi" for r in rows)
    assert len(rows) == 2


def test_tags_any_match_and_normalized(sqlite_api):
    _seed(sqlite_api)
    # Upper-case + spacing normalize to the stored labels; ANY-match across the list.
    rows = browse_memories(tags=["PLL", "pgvector"])
    summaries = {r["summary"] for r in rows}
    assert summaries == {
        "The PLL loop filter reduces phase noise",
        "Adopt pgvector for semantic recall",
    }


def test_authorship_split_out_of_tags(sqlite_api):
    _seed(sqlite_api)
    row = next(r for r in browse_memories(persona="awsi") if "pgvector" in r["tags"])
    assert row["by"] == "abdu"  # by:<slug> separated into `by`
    assert all(not t.startswith("by:") for t in row["tags"])  # not leaked into tags


def test_occurred_range(sqlite_api):
    _seed(sqlite_api)
    after = browse_memories(occurred_after="2026-07-22")
    assert {r["summary"] for r in after} == {"Adopt pgvector for semantic recall"}
    before = browse_memories(occurred_before="2026-07-20")
    assert {r["summary"] for r in before} == {"The PLL loop filter reduces phase noise"}


def test_min_hit_count(sqlite_api):
    _seed(sqlite_api)
    recall.recall_facts("awsi", "pgvector semantic", k=5)  # bump one row's hit
    rows = browse_memories(persona="awsi", min_hit_count=1)
    assert [r["summary"] for r in rows] == ["Adopt pgvector for semantic recall"]


def test_scope_own_team_all(sqlite_api, monkeypatch):
    """own/team is the USER axis: seed a SECOND human (different MEMORY_USER, same
    namespace-fixed project) so ``team`` is non-empty."""
    _seed(sqlite_api)  # authored by MEMORY_USER=abdu
    monkeypatch.setenv("MEMORY_USER", "teammate")
    recall.store_facts_verbatim("qiyas", [("Teammate ran the benchmark", ["bench"], None)])
    monkeypatch.setenv("MEMORY_USER", "abdu")  # browse AS abdu

    own = browse_memories(scope="own")
    team = browse_memories(scope="team")
    all_ = browse_memories(scope="all")
    assert {r["agent_type"] for r in own} == {"awsi", "dina"}  # abdu's personas
    assert all(r["scope"] == "own" for r in own)
    assert {r["summary"] for r in team} == {"Teammate ran the benchmark"}
    assert all(r["scope"] == "team" for r in team)
    assert len(all_) == 4  # own (3) + team (1)


def test_persona_and_scope_orthogonal(sqlite_api):
    _seed(sqlite_api)
    # persona filters the role axis independently of scope.
    rows = browse_memories(persona="awsi", scope="all")
    assert {r["agent_type"] for r in rows} == {"awsi"}
    assert len(rows) == 2


# ── lexical text ──────────────────────────────────────────────────────────────
def test_lexical_text_filter(sqlite_api):
    _seed(sqlite_api)
    rows = browse_memories(text="phase noise")
    assert {r["summary"] for r in rows} == {"The PLL loop filter reduces phase noise"}


def test_lexical_text_no_match(sqlite_api):
    _seed(sqlite_api)
    assert browse_memories(text="quantum entanglement") == []


# ── ordering ──────────────────────────────────────────────────────────────────
def test_order_occurred_desc_nulls_last(sqlite_api):
    _seed(sqlite_api)
    recall.store_facts_verbatim("awsi", [("undated fact", [], None)])
    rows = browse_memories(order="occurred")
    dates = [r["occurred_at"] for r in rows]
    non_null = [d for d in dates if d is not None]
    assert non_null == sorted(non_null, reverse=True)  # occurred desc
    assert dates[-1] is None  # NULL occurred sorts last


def test_order_hits_desc(sqlite_api):
    _seed(sqlite_api)
    recall.recall_facts("awsi", "pgvector semantic", k=5)  # only this row gets a hit
    rows = browse_memories(persona="awsi", order="hits")
    assert rows[0]["summary"] == "Adopt pgvector for semantic recall"
    assert rows[0]["hit_count"] >= 1


def test_order_recency_is_created_desc(sqlite_api):
    _seed(sqlite_api)
    created = [r["created_at"] for r in browse_memories()]
    assert created == sorted(created, reverse=True)  # non-increasing by created_at


# ── pagination ────────────────────────────────────────────────────────────────
def test_limit_and_offset_paginate(sqlite_api):
    _seed(sqlite_api)
    full = browse_memories(order="occurred")
    assert len(full) == 3
    page1 = browse_memories(order="occurred", limit=2, offset=0)
    page2 = browse_memories(order="occurred", limit=2, offset=2)
    assert [r["id"] for r in page1] == [r["id"] for r in full[:2]]
    assert [r["id"] for r in page2] == [r["id"] for r in full[2:]]


# ── READ-ONLY invariant (the important one) ─────────────────────────────────────
def test_browse_never_bumps_hit_count(sqlite_api):
    _seed(sqlite_api)
    recall.recall_facts("awsi", "pgvector semantic", k=5)  # establish hit_count = 1

    def _hits() -> dict[str, int]:
        return {r["summary"]: r["hit_count"] for r in browse_memories(persona="awsi")}

    before = _hits()
    assert before["Adopt pgvector for semantic recall"] == 1
    for _ in range(3):  # browse repeatedly...
        browse_memories(persona="awsi", text="pgvector semantic")
        browse_memories(persona="awsi", order="hits")
    assert _hits() == before  # ...hit_count is byte-identical (no writeback)


def test_browse_issues_no_write(sqlite_api):
    _seed(sqlite_api)
    Session = sessionmaker(bind=sqlite_api, future=True)
    with Session() as s:
        last_used_before = {
            m.id: m.last_used_at for m in s.query(AgentMemory).all()
        }
    browse_memories(text="pgvector", order="hits")
    with Session() as s:
        last_used_after = {m.id: m.last_used_at for m in s.query(AgentMemory).all()}
    assert last_used_after == last_used_before  # browse touched no row


# ── validation ──────────────────────────────────────────────────────────────────
def test_bad_scope_and_order_raise(sqlite_api):
    with pytest.raises(ValueError):
        browse_memories(scope="everyone")
    with pytest.raises(ValueError):
        browse_memories(order="alphabetical")


# ── semantic degrade on SQLite (no pgvector) ─────────────────────────────────────
def test_semantic_degrades_to_lexical_on_sqlite(sqlite_api):
    _seed(sqlite_api)
    # No pgvector on SQLite -> semantic falls back to the lexical filter + order.
    rows = browse_memories(text="phase noise", semantic=True)
    assert {r["summary"] for r in rows} == {"The PLL loop filter reduces phase noise"}


# ── semantic on real Postgres + pgvector (DB-gated) ──────────────────────────────
def test_semantic_ranks_by_cosine_and_stays_read_only(db_session, monkeypatch):
    """Real Postgres + pgvector. Skips without a DB; FAILS if the vector leg is
    silently absent when Postgres is up."""
    monkeypatch.setenv("MEMORY_USER", "abdu")
    fake = _FakeEmbedder()
    # Store writes embeddings via memory.store; browse embeds the query via
    # memory.browse — patch BOTH to the same fake so the vectors share one space.
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _p: fake)
    monkeypatch.setattr("memory.browse.get_embedding_service", lambda _p: fake)

    recall.store_facts_verbatim(
        "awsi",
        [
            ("Adopt pgvector for semantic recall", ["pgvector"], None),
            ("The weekly standup moved to Tuesday", ["standup"], None),
        ],
    )
    target = "Adopt pgvector for semantic recall"
    rows = browse_memories(persona="awsi", text=target, semantic=True)
    assert rows, "semantic browse returned nothing — vector leg silently absent"
    assert rows[0]["summary"] == target  # exact text -> cosine distance 0 -> ranks first

    hits_before = {r["summary"]: r["hit_count"] for r in browse_memories(persona="awsi")}
    browse_memories(persona="awsi", text=target, semantic=True)
    hits_after = {r["summary"]: r["hit_count"] for r in browse_memories(persona="awsi")}
    assert hits_after == hits_before  # semantic browse is read-only too
    assert all(v == 0 for v in hits_after.values())


# ── full cell: browse rows + get_memory share one materializer ───────────────────
def test_browse_rows_carry_full_cell(sqlite_api):
    _seed(sqlite_api)
    rows = browse_memories(persona="awsi")
    assert rows and all(set(r) == _FULL_CELL_KEYS for r in rows)


def test_get_memory_returns_full_cell(sqlite_api):
    _seed(sqlite_api)
    row = next(r for r in browse_memories(persona="awsi") if "pgvector" in r["tags"])
    cell = get_memory(row["id"])
    assert cell is not None
    assert set(cell) == _FULL_CELL_KEYS  # EVERY column present
    assert cell["id"] == row["id"]
    assert cell["content"] == "Adopt pgvector for semantic recall"
    assert cell["agent_type"] == "awsi"
    assert cell["scope"] == "own"
    assert cell["by"] == "abdu"
    assert "pgvector" in cell["tags"]
    assert all(not t.startswith("by:") for t in cell["tags"])  # authorship split out
    assert cell["is_active"] is True
    assert cell["memory_tier"] == "agent_working"
    assert cell["salience"] == 0.0
    # SQLite degraded path: no vector, no tsv — presence flags reflect that.
    assert cell["has_embedding"] is False
    assert cell["embedding_dim"] is None
    assert cell["has_tsv"] is False


def test_get_memory_not_found_returns_none(sqlite_api):
    _seed(sqlite_api)
    assert get_memory(str(uuid.uuid4())) is None  # well-formed but absent


def test_get_memory_bad_id_returns_none(sqlite_api):
    _seed(sqlite_api)
    assert get_memory("not-a-uuid") is None
    assert get_memory("") is None


def test_get_memory_out_of_scope_returns_none(sqlite_api):
    """A cell in a DIFFERENT project pool is invisible to get_memory (scope-gated)."""
    _seed(sqlite_api)
    foreign_id = uuid.uuid4()
    Session = sessionmaker(bind=sqlite_api, future=True)
    with Session() as s:
        s.add(
            AgentMemory(
                id=foreign_id,
                content="a cell owned by another project",
                agent_type="awsi",
                user_id=uuid.uuid4(),
                project_id=uuid.uuid4(),  # not this replica's project pool
                customer_id=uuid.uuid4(),
            )
        )
        s.commit()
    assert get_memory(str(foreign_id)) is None


def test_get_memory_is_read_only(sqlite_api):
    """Reading the cell must NOT bump hit_count / last_used_at / salience."""
    _seed(sqlite_api)
    recall.recall_facts("awsi", "pgvector semantic", k=5)  # establish hit_count = 1
    row = next(r for r in browse_memories(persona="awsi") if "pgvector" in r["tags"])
    before = get_memory(row["id"])
    assert before["hit_count"] == 1
    for _ in range(3):
        get_memory(row["id"])  # read repeatedly...
    after = get_memory(row["id"])
    assert after["hit_count"] == 1  # ...count is byte-identical
    assert after["last_used_at"] == before["last_used_at"]
    assert after["salience"] == before["salience"]


# ── embedding presence is DB-gated (real Postgres + pgvector) ────────────────────
def test_get_memory_reports_embedding_presence_on_postgres(db_session, monkeypatch):
    """Skips without a DB; on real Postgres asserts the vector/tsv PRESENCE flags and
    the 384-d dim — without ever dumping the raw floats — and stays read-only."""
    monkeypatch.setenv("MEMORY_USER", "abdu")
    fake = _FakeEmbedder()
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _p: fake)
    recall.store_facts_verbatim(
        "awsi", [("Adopt pgvector for semantic recall", ["pgvector"], None)]
    )
    row = next(r for r in browse_memories(persona="awsi") if "pgvector" in r["tags"])
    cell = get_memory(row["id"])
    assert cell is not None
    assert cell["has_embedding"] is True
    assert cell["embedding_dim"] == config.EMBEDDING_DIM  # 384, not the raw payload
    assert "embedding" not in cell  # raw floats never leak
    assert cell["has_tsv"] is True  # content_tsv populated on PG
    before = cell["hit_count"]
    get_memory(row["id"])  # read-only on PG too
    assert get_memory(row["id"])["hit_count"] == before
