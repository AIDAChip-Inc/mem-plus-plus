"""Project boundaries without disabling organizational team memory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import run_benchmark
from eval.memory_client import MemoryClient
from memory import config, recall
from memory.models import Base


@pytest.fixture(params=["sqlite", "postgres"])
def client(tmp_path, monkeypatch, request):
    dsn = os.environ.get("MEMORY_DATABASE_URL")
    schema = None
    admin = None
    if request.param == "postgres":
        if not dsn or not dsn.startswith("postgresql"):
            pytest.skip("Postgres scope proof needs MEMORY_DATABASE_URL")
        admin = create_engine(dsn)
        schema = "scope_test_" + uuid.uuid4().hex
        with admin.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = admin.execution_options(schema_translate_map={None: schema})
        class Embedder:
            def embed(self, text, *, intent="document"):
                return [1.0] + [0.0] * 383
        monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Embedder())
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'scope.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr("memory.db._SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setenv("MEMORY_USER", "scope-reviewer")
    monkeypatch.setattr(config, "MEMORY_PROJECT_ENABLED", True)
    try:
        yield MemoryClient()
    finally:
        if schema is not None:
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


def summaries(rows):
    return {r["summary"] for r in rows}


def test_org_sharing_survives_and_other_examples_are_excluded(client, monkeypatch):
    with client.example_scope("org", "alpha"):
        client.store_facts_verbatim("engineering", ["Atlas release approved by engineering"])
        monkeypatch.setenv("MEMORY_USER", "another-contributor")
        client.store_facts_verbatim("finance", ["Atlas release approved by finance"])
    with client.example_scope("org", "beta"):
        client.store_facts_verbatim("legal", ["Atlas release blocked in beta"])
    with client.example_scope("org", "alpha"):
        rows = client.recall_facts("coordinator", "Atlas release", k=10, update_hits=False)
        assert summaries(rows) == {"Atlas release approved by engineering",
                                   "Atlas release approved by finance"}
        assert client.count_facts("finance") == 1
    with client.example_scope("other-benchmark", "alpha"):
        assert client.recall_facts("coordinator", "Atlas", update_hits=False) == []
        assert client.count_facts("finance") == 0


def test_nested_scopes_restore_defaults_after_exception(client):
    default = recall._scope()
    with client.example_scope("bench", "a"):
        outer = recall._scope()
        with pytest.raises(RuntimeError):
            with client.example_scope("bench", "b"):
                assert recall._scope()[1:3] != outer[1:3]
                raise RuntimeError("test")
        assert recall._scope() == outer
    assert recall._scope() == default


def test_contexts_are_thread_local(client):
    from threading import Barrier
    barrier = Barrier(2)
    def worker(example):
        with client.example_scope("bench", example):
            before = recall._scope()
            barrier.wait(timeout=5)
            assert recall._scope() == before
            return before[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(worker, ["a", "b"]))
    assert a != b


def test_real_harness_parallel_reuse_and_dates(client):
    convs = [Conversation(f"c{i}", [Turn(f"t{i}", "author", f"Project secret is code{i}",
             datetime(2024, 1, 1, tzinfo=UTC))],
             [QAItem(f"q{i}", "Project secret?", [f"code{i}"], "single_hop", {f"t{i}"},
              question_date=datetime(2024, 2, 1, tzinfo=UTC))]) for i in range(2)]
    serial = run_benchmark(convs, client, benchmark="scope-proof", recall_only=True,
                           count_tokens=False)
    parallel = run_benchmark(convs, client, benchmark="scope-proof", recall_only=True,
                             count_tokens=False, workers=2, reuse_ingest=True)
    assert serial["config"]["memory_scope_policy"] == client.scope_policy
    assert parallel["config"]["n_conversations_reused"] == 2
    assert serial["config"]["recall_params"]["project_memory_enabled"] is True
    for result in (serial, parallel):
        for i, row in enumerate(result["results"]):
            assert row["retrieved_turn_ids"] == [f"t{i}"]
    # Verify absence directly too: turn-ID scoring would ignore foreign text.
    for i in range(2):
        with client.example_scope("scope-proof", f"c{i}"):
            assert summaries(client.recall_facts(f"researcher:c{i}", "Project secret", k=10,
                                                update_hits=False)) == {
                convs[i].turns[0].render()}


def test_scope_ids_do_not_collide_on_delimiters(client):
    with client.example_scope("a:b", "c"):
        first = recall._scope()[1:3]
    with client.example_scope("a", "b:c"):
        assert recall._scope()[1:3] != first


def test_consolidation_sweep_uses_same_project(client, monkeypatch):
    from eval.consolidation_eval import build_default_corpus, run_consolidation_eval
    from memory.consolidation import consolidate
    monkeypatch.setattr("memory.consolidation.llm_available", lambda: False)
    corpus = build_default_corpus()
    seen = []
    def checked_consolidate(persona, **kwargs):
        assert client.count_facts(persona) == len(set(corpus.facts))
        seen.append(recall._scope()[1])
        return consolidate(persona, **kwargs)
    result = run_consolidation_eval(memory=client, consolidate_fn=checked_consolidate,
                                   corpus=corpus, taus=(0.8,), persona_base="scope-consol",
                                   n_runs=1, backend="engine")
    persona = "scope-consol__t0.8__r0"
    with client.example_scope("consolidation_quality", persona):
        assert seen == [recall._scope()[1]]
        assert client.count_facts(persona) == result["per_tau"][0]["memories_after"]
    assert client.count_facts(persona) == 0
