"""Always-runnable E2E: clean imports + the SQLite-degraded public-API flow.

No DB, no network, no model. Proves the ``memory.recall`` public surface
(``store_facts_verbatim`` / ``store_facts`` / ``recall_facts``) runs end-to-end
on the degraded path, that lexical+tag ranking + hit writeback behave, and that
the vector leg is CORRECTLY ABSENT (not silently faked) on SQLite. The full
ONNX+pgvector semantic leg is proven in ``test_e2e_pgvector.py`` (DB-gated).
"""
from __future__ import annotations

import importlib

import pytest

from memory import recall


def test_all_modules_import_cleanly():
    """Every shipped module imports without side-effects — the cheapest seam
    check (a broken import breaks a clean clone before any test runs)."""
    for mod in (
        "memory.config", "memory.db", "memory.models", "memory.store",
        "memory.recall", "memory.extraction", "memory.llm",
        "memory.embeddings", "memory.embeddings.factory",
        "memory.embeddings.onnx_service", "memory.embeddings.provider",
        "eval.harness", "eval.metrics", "eval.memory_client", "eval.judge",
        "eval.llm", "eval._runner", "eval.adapters.base", "eval.adapters.locomo",
        "eval.adapters.longmemeval", "eval.adapters.longbench",
    ):
        assert importlib.import_module(mod) is not None


def test_verbatim_then_recall_ranks_relevant_first(sqlite_public_api):
    """store_facts_verbatim → recall_facts: the relevance-matched fact leads and
    accrues exactly one hit; the by:<slug> authorship tag surfaces."""
    res = recall.store_facts_verbatim(
        "awsi",
        [
            ("Adopt pgvector for semantic recall over embeddings", ["pgvector"], None),
            ("The weekly standup moved to Tuesday morning", ["standup"], None),
        ],
    )
    assert res["written"] == 2 and res["mode"] == "facts"

    rows = recall.recall_facts("awsi", "pgvector recall", k=5)
    assert rows, "recall returned nothing on the degraded path"
    assert "pgvector" in rows[0]["summary"].lower()
    assert rows[0]["hit_count"] == 1  # only the matched row accrues a hit
    assert rows[0]["by"] == "burhan-e2e"  # deterministic contributor slug


def test_recall_hit_count_accumulates_across_calls(sqlite_public_api):
    """Hit writeback is durable: repeated relevance-matched recalls increment."""
    recall.store_facts_verbatim("awsi", [("Kafka message bus for agents", ["kafka"], None)])
    recall.recall_facts("awsi", "kafka message bus", k=5)
    rows = recall.recall_facts("awsi", "kafka message bus", k=5)
    top = next(r for r in rows if "kafka" in r["summary"].lower())
    assert top["hit_count"] == 2


def test_exact_summary_dedup_is_idempotent(sqlite_public_api):
    """Re-storing an identical summary in-scope writes nothing (retry-safe)."""
    recall.store_facts_verbatim("burhan", [("a unique dedup fact", [], None)])
    again = recall.store_facts_verbatim("burhan", [("a unique dedup fact", [], None)])
    assert again["written"] == 0


def test_store_facts_degrades_without_llm(sqlite_public_api):
    """No credential in this env → the gated auto path degrades visibly and
    stores nothing (never a silent success)."""
    res = recall.store_facts("awsi", "User said the tapeout slips to Q3.")
    assert res["written"] == 0
    assert res["mode"] == "gated"
    assert res.get("degraded") is True


def test_store_facts_empty_is_reported(sqlite_public_api):
    assert recall.store_facts("awsi", "   ")["reason"] == "empty"


def test_vector_leg_is_absent_on_sqlite(sqlite_public_api):
    """On SQLite the vector leg is STRUCTURALLY absent (PG-only in the store) —
    this is the degraded path, and it must be honestly empty, not faked."""
    from sqlalchemy.orm import sessionmaker

    from memory.models import AgentMemory
    from memory.store import PostgresMemoryStore

    session = sessionmaker(bind=sqlite_public_api, future=True)()
    try:
        store = PostgresMemoryStore(session)
        assert store._is_postgres is False
        base = session.query(AgentMemory)
        assert store._vector_candidates(base, "anything") == []
    finally:
        session.close()
