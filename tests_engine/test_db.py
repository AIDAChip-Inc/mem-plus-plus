"""DB-backed tests — real Postgres + pgvector, fake (deterministic) embedder.

Skipped when MEMORY_DATABASE_URL is unset/unreachable (see conftest).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from memory import config, recall
from memory.models import AgentMemory
from memory.store import PostgresMemoryStore

_U = uuid.uuid5(uuid.NAMESPACE_DNS, "test-user")
_P = uuid.uuid5(uuid.NAMESPACE_DNS, "test-project")
_C = uuid.uuid5(uuid.NAMESPACE_DNS, "test-customer")


def _write(store, *, summary, tags=None, agent="awsi", user=_U, occurred_at=None):
    return store.write(
        user_id=user,
        agent_type=agent,
        project_id=_P,
        customer_id=_C,
        content=summary,
        context_summary=summary,
        tags=tags or [],
        occurred_at=occurred_at,
    )


def test_write_persists_embedding_and_tsv(db_session):
    store = PostgresMemoryStore(db_session)
    m = _write(store, summary="The PLL loop filter reduces phase noise", tags=["pll", "phase-noise"])
    row = db_session.get(AgentMemory, m.id)
    assert row.embedding is not None and len(list(row.embedding)) == config.EMBEDDING_DIM
    assert row.content_tsv is not None
    assert {t.label for t in row.tags} == {"pll", "phase-noise"}


def test_recall_ranks_relevant_above_recent_and_writes_hit(db_session):
    store = PostgresMemoryStore(db_session)
    hit = _write(store, summary="Adopt pgvector for semantic recall", tags=["pgvector"])
    _write(store, summary="Weekly standup moved to Tuesday", tags=["standup"])
    rows = store.recall(user_id=_U, agent_type="awsi", project_id=_P, query="pgvector recall", k=5)
    assert rows[0].id == hit.id  # relevance-matched row leads
    # Only the relevance-matched row accrues a hit.
    refreshed = db_session.get(AgentMemory, hit.id)
    db_session.refresh(refreshed)
    assert refreshed.hit_count == 1
    assert refreshed.last_hit_by == "awsi"


def test_matched_salience_dominates_band(db_session):
    store = PostgresMemoryStore(db_session)
    _write(store, summary="Kafka message bus for agent coordination", tags=["kafka"])
    _, plan = store.recall_with_plan(
        user_id=_U, agent_type="awsi", project_id=_P, query="kafka message bus", k=5
    )
    matched = [sal for _mid, sal, is_match in plan if is_match]
    assert matched and min(matched) >= config.SALIENCE_MATCH_BAND


def test_idempotent_write_on_source_message_id(db_session):
    store = PostgresMemoryStore(db_session)
    smid = uuid.uuid4()
    first = store.write(
        user_id=_U, agent_type="awsi", project_id=_P, customer_id=_C,
        content="idempotent", context_summary="idempotent", source_message_id=smid,
    )
    second = store.write(
        user_id=_U, agent_type="awsi", project_id=_P, customer_id=_C,
        content="idempotent", context_summary="idempotent", source_message_id=smid,
    )
    assert first.id == second.id


def test_project_sections_team_is_disjoint_from_own(db_session):
    store = PostgresMemoryStore(db_session)
    mine = _write(store, summary="Awsi built the recall engine", agent="awsi", tags=["engine"])
    theirs = _write(store, summary="Dina designed the recall engine schema", agent="dina", tags=["engine"])
    own, _op, team, _tp = store.recall_project_sections(
        user_id=_U, agent_type="awsi", project_id=_P, customer_id=_C, query="recall engine", k=10
    )
    own_ids = {r.id for r in own}
    team_ids = {r.id for r in team}
    assert mine.id in own_ids
    assert theirs.id in team_ids
    assert own_ids.isdisjoint(team_ids)  # sections disjoint by construction


def test_occurred_before_after_filter(db_session):
    store = PostgresMemoryStore(db_session)
    old = _write(store, summary="event alpha happened", occurred_at=datetime(2020, 1, 1, tzinfo=UTC))
    _write(store, summary="event beta happened", occurred_at=datetime(2025, 1, 1, tzinfo=UTC))
    rows = store.recall(
        user_id=_U, agent_type="awsi", project_id=_P, query="event",
        occurred_before=datetime(2021, 1, 1, tzinfo=UTC), k=10, update_hits=False,
    )
    assert {r.id for r in rows} == {old.id}


# ── recall.py public API ───────────────────────────────────────────────────────
def test_store_facts_verbatim_then_recall_facts(db_session, monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "alice")
    res = recall.store_facts_verbatim(
        "awsi",
        [
            ("The engine ports store.py near-verbatim", ["store", "port"], None),
            ("RRF weights are fuzzy/tag/vector = 1/1/4", ["rrf"], datetime(2026, 7, 23, tzinfo=UTC)),
        ],
    )
    assert res["written"] == 2 and res["mode"] == "facts"
    rows = recall.recall_facts("awsi", "RRF weights", k=10)
    assert any("RRF weights" in r["summary"] for r in rows)
    top = next(r for r in rows if "RRF weights" in r["summary"])
    assert top["by"] == "alice"  # authorship tag surfaced
    assert top["occurred_at"] == "2026-07-23"
    assert isinstance(top["hit_count"], int)


def test_store_facts_verbatim_dedups_exact_summary(db_session, monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "alice")
    recall.store_facts_verbatim("burhan", [("a unique dedup fact", [], None)])
    again = recall.store_facts_verbatim("burhan", [("a unique dedup fact", [], None)])
    assert again["written"] == 0  # exact-summary dedup in-scope


def test_store_facts_degrades_without_llm(db_session):
    # No LLM available in the test env -> gated path degrades, stores nothing.
    res = recall.store_facts("awsi", "User said the tapeout slips to Q3.")
    assert res["written"] == 0
    assert res["mode"] == "gated"  # auto -> gated (MEMORY_ATOMIC_FACTS default OFF)
    assert res.get("degraded") is True


def test_store_facts_empty_content():
    assert recall.store_facts("awsi", "   ")["reason"] == "empty"
