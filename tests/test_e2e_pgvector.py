"""DB-gated: the real Postgres 16 + pgvector + ONNX semantic path.

This is the part Awsi flagged as UNPROVEN — it needs a container runtime. These
tests SKIP (see ``pg_engine`` / ``real_embedder`` fixtures) when the DB or the
model is absent, and FAIL LOUDLY when the DB is present but the behavior is wrong.

The headline is ``test_semantic_only_recall_fires_the_vector_leg`` plus its A/B
control: a fact that shares NO lexical token and NO tag with the query is
retrieved and relevance-ranked ONLY because the ONNX embedding + pgvector cosine
leg fired. If that leg were silently degraded (the failure Awsi worried about),
the fact would be invisible and the test would FAIL — never pass vacuously.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from memory import config
from memory.models import AgentMemory
from memory.store import PostgresMemoryStore

# Every test here needs BOTH a live Postgres AND the real embedder.
pytestmark = [pytest.mark.db, pytest.mark.embedder]


def _session(engine):
    return sessionmaker(bind=engine, expire_on_commit=True, future=True)()


def _write(store, *, summary, tags=None, agent="awsi", user=None, project=None, customer=None):
    return store.write(
        user_id=user,
        agent_type=agent,
        project_id=project,
        customer_id=customer,
        content=summary,
        context_summary=summary,
        tags=tags or [],
    )


def test_write_persists_a_real_minilm_embedding(pg_engine, real_embedder):
    """The write path stores a genuine 384-d MiniLM vector (PG only) — and it is
    the SAME vector the embedder produces for that text, not a placeholder."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        assert store._is_postgres is True
        text = "The PLL loop filter reduces phase noise near the carrier"
        m = _write(store, summary=text, tags=["pll", "phase-noise"], user=u, project=p, customer=c)
        row = session.get(AgentMemory, m.id)
        assert row.content_tsv is not None
        stored = np.asarray(list(row.embedding), dtype=np.float32)
        assert stored.shape == (config.EMBEDDING_DIM,)
        assert abs(np.linalg.norm(stored) - 1.0) < 1e-3  # a real unit vector
        expected = np.asarray(real_embedder.embed(text), dtype=np.float32)
        assert float(stored @ expected) > 0.999  # identical up to fp rounding
    finally:
        session.close()


def test_rrf_orders_relevant_first_and_writes_hit(pg_engine, real_embedder):
    """Full recall: the relevance-matched row leads and accrues exactly one hit."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        hit = _write(store, summary="Adopt pgvector for semantic recall", tags=["pgvector"],
                     user=u, project=p, customer=c)
        _write(store, summary="Weekly standup moved to Tuesday", tags=["standup"],
               user=u, project=p, customer=c)
        rows = store.recall(user_id=u, agent_type="awsi", project_id=p, query="pgvector recall", k=5)
        assert rows[0].id == hit.id
        refreshed = session.get(AgentMemory, hit.id)
        session.refresh(refreshed)
        assert refreshed.hit_count == 1
        assert refreshed.last_hit_by == "awsi"
    finally:
        session.close()


def test_semantic_only_recall_fires_the_vector_leg(pg_engine, real_embedder):
    """PROOF the vector leg fires: a fact with NO lexical token and NO tag in
    common with the query is retrieved and relevance-matched — only cosine
    similarity over the ONNX embedding can explain it."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        # "canine companion greeted its owner" shares no stem/tag with the query.
        target = _write(
            store,
            summary="A canine companion greeted its owner with great enthusiasm",
            tags=["household-pet"], user=u, project=p, customer=c,
        )
        _write(store, summary="The quarterly budget spreadsheet was finalized",
               tags=["finance"], user=u, project=p, customer=c)

        query = "dog welcoming behaviour toward humans"

        # Direct leg check: the vector candidate set must contain the target.
        base = session.query(AgentMemory).filter(
            AgentMemory.user_id == u, AgentMemory.agent_type == "awsi",
            AgentMemory.project_id == p, AgentMemory.is_active.is_(True),
        )
        vec_ids = {m.id for m in store._vector_candidates(base, query)}
        assert target.id in vec_ids, "vector leg did not surface the semantic neighbor"

        # End-to-end: the target is recalled AND relevance-matched (rrf > 0).
        rows, plan = store.recall_with_plan(
            user_id=u, agent_type="awsi", project_id=p, query=query, k=5
        )
        assert target.id in {r.id for r in rows}, "semantic neighbor not recalled"
        matched = {mid for mid, _sal, is_match in plan if is_match}
        assert target.id in matched, (
            "semantic neighbor recalled but NOT relevance-matched — the vector leg "
            "is silently absent (degraded to lexical+tag)"
        )
        # Cosine ranks the semantic neighbor first (offline-verified separation:
        # cos(query,target)=0.71 vs cos(query,distractor)=-0.05).
        assert rows[0].id == target.id, "vector leg fired but did not rank the neighbor first"
    finally:
        session.close()


def test_vector_leg_is_the_cause_ab_control(pg_engine, real_embedder, monkeypatch):
    """A/B control isolating the leg: with MEMORY_EMBEDDINGS_ENABLED OFF, the same
    lexically/tag-disjoint fact is NO LONGER a relevance match. Proves the match
    in the previous test came from the vector leg, not some other signal."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        target = _write(
            store,
            summary="A canine companion greeted its owner with great enthusiasm",
            tags=["household-pet"], user=u, project=p, customer=c,
        )
        query = "dog welcoming behaviour toward humans"

        monkeypatch.setattr(config, "MEMORY_EMBEDDINGS_ENABLED", False)
        assert store._vector_candidates(
            session.query(AgentMemory).filter(AgentMemory.project_id == p), query
        ) == []  # read leg is a true no-op when the flag is off

        _rows, plan = store.recall_with_plan(
            user_id=u, agent_type="awsi", project_id=p, query=query, k=5
        )
        matched = {mid for mid, _sal, is_match in plan if is_match}
        assert target.id not in matched, (
            "fact still relevance-matched with embeddings OFF — the earlier match "
            "was not attributable to the vector leg"
        )
    finally:
        session.close()


def test_matched_salience_dominates_band(pg_engine, real_embedder):
    """A relevance-matched row's salience sits in the matched band (dominates the
    hits+recency terms), preserved exactly from production."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        _write(store, summary="Kafka message bus for agent coordination", tags=["kafka"],
               user=u, project=p, customer=c)
        _rows, plan = store.recall_with_plan(
            user_id=u, agent_type="awsi", project_id=p, query="kafka message bus", k=5
        )
        matched = [sal for _mid, sal, is_match in plan if is_match]
        assert matched and min(matched) >= config.SALIENCE_MATCH_BAND
    finally:
        session.close()


def test_two_section_recall_team_disjoint_from_own(pg_engine, real_embedder):
    """Project-memory two-section split: OWN and TEAM sections are disjoint by
    construction over the real DB."""
    u, p, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _session(pg_engine)
    try:
        store = PostgresMemoryStore(session)
        mine = _write(store, summary="Awsi built the recall engine", agent="awsi",
                      tags=["engine"], user=u, project=p, customer=c)
        theirs = _write(store, summary="Dina designed the recall engine schema", agent="dina",
                        tags=["engine"], user=u, project=p, customer=c)
        own, _op, team, _tp = store.recall_project_sections(
            user_id=u, agent_type="awsi", project_id=p, customer_id=c,
            query="recall engine", k=10,
        )
        own_ids = {r.id for r in own}
        team_ids = {r.id for r in team}
        assert mine.id in own_ids
        assert theirs.id in team_ids
        assert own_ids.isdisjoint(team_ids)
    finally:
        session.close()


def test_public_recall_facts_end_to_end_on_postgres(pg_public_api, real_embedder):
    """The public API (recall.store_facts_verbatim → recall.recall_facts) over the
    REAL engine + Postgres: a semantic query with no lexical overlap still recalls
    the fact — the full ONNX+pgvector path Awsi could not run."""
    from memory import recall

    recall.store_facts_verbatim(
        "burhan",
        [
            ("A canine companion greeted its owner with great enthusiasm", ["household-pet"], None),
            ("The quarterly budget spreadsheet was finalized", ["finance"], None),
        ],
    )
    rows = recall.recall_facts("burhan", "dog welcoming behaviour toward humans", k=5)
    assert any("canine companion" in r["summary"] for r in rows), (
        "public recall_facts did not surface the semantic neighbor over Postgres"
    )
