"""HTTP contract, atomic retries, and independent AML memory worlds."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from memory.aml_api import AMLService, _NS, _identity
from memory.models import AgentMemory, Base, MemoryAPIRequest

TOKEN = "local-test-token-not-a-real-secret-0123456789"


@pytest.fixture(params=["sqlite", "postgres"])
def api(tmp_path, monkeypatch, request):
    schema = None
    if request.param == "postgres":
        dsn = os.environ.get("MEMORY_DATABASE_URL", "")
        if not dsn.startswith("postgresql"):
            pytest.skip("Postgres API tests require MEMORY_DATABASE_URL")
        # Exercise UTC serialization even when the database session is non-UTC.
        admin = create_engine(dsn, connect_args={"options": "-c timezone=America/Los_Angeles"})
        schema = "aml_test_" + uuid.uuid4().hex
        with admin.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = admin.execution_options(schema_translate_map={None: schema})
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    try:
        Base.metadata.create_all(engine)
        if request.param == "sqlite":
            def populate(self, row, **kwargs):
                row.embedding = [1.0] + [0.0] * 383
            monkeypatch.setattr("memory.store.PostgresMemoryStore._populate_search_fields", populate)
        else:
            class Embedder:
                def embed(self, text, *, intent="document"):
                    return [1.0] + [0.0] * 383
            monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Embedder())
        sessions = sessionmaker(bind=engine)
        yield AMLService(sessions, TOKEN), sessions
    finally:
        if schema is not None:
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


def invoke(service, path, payload=None, *, auth=True, method="POST", **overrides):
    raw = json.dumps(payload).encode("utf-8") if payload is not None else b""
    env = {"PATH_INFO": path, "REQUEST_METHOD": method,
           "CONTENT_TYPE": "application/json", "CONTENT_LENGTH": str(len(raw)),
           "wsgi.input": BytesIO(raw)}
    if auth:
        env["HTTP_AUTHORIZATION"] = "Bearer " + TOKEN
    env.update(overrides)
    seen = []
    body = b"".join(service(env, lambda status, headers: seen.append((status, headers))))
    return int(seen[0][0].split()[0]), json.loads(body)


def chunk(user="alpha", request="r1", session="s1", content="Atlas approved"):
    return {"request_id": request, "user_id": user, "session_id": session,
            "messages": [{"role": "user", "content": content, "timestamp": 1704067200000}]}


def test_auth_and_contract(api):
    service, _ = api
    assert invoke(service, "/health", auth=False, method="GET")[0] == 200
    assert invoke(service, "/add", chunk(), auth=False)[0] == 401
    assert invoke(service, "/add", chunk(), HTTP_AUTHORIZATION="Bearer wrong")[0] == 401
    assert invoke(service, "/add", chunk(), CONTENT_TYPE="text/plain")[0] == 415
    assert invoke(service, "/add", {**chunk(), "unexpected": True})[0] == 422
    assert invoke(service, "/search", {"user_id": "a", "query": "Atlas", "top_k": True})[0] == 422
    assert invoke(service, "/search", {"user_id": "a", "query": "Atlas", "top_k": 101})[0] == 422
    assert invoke(service, "/search", {"user_id": "a", "query": " ", "top_k": 10})[0] == 422
    assert invoke(service, "/add", chunk(content=" "))[0] == 422
    bad = chunk(); bad["messages"][0]["timestamp"] = 10**30
    assert invoke(service, "/add", bad)[0] == 422


def test_sources_share_sessions_but_never_other_worlds(api):
    service, sessions = api
    for payload in [chunk(), chunk(request="r2", session="s2", content="Atlas finance approved"),
                    chunk(user="beta", content="Atlas blocked in unrelated world")]:
        status, response = invoke(service, "/add", payload)
        assert status == 200
        assert response == {"success": True, "request_id": payload["request_id"],
                            "session_id": payload["session_id"], "user_id": payload["user_id"]}
    query = {"user_id": "alpha", "query": "Atlas", "top_k": 100,
             "options": ["irrelevant multiple-choice option"]}
    status, response = invoke(service, "/search", query)
    assert status == 200
    rows = response["data"]
    assert {r["content"] for r in rows} == {"[2024-01-01T00:00:00+00:00] user: Atlas approved",
                                            "[2024-01-01T00:00:00+00:00] user: Atlas finance approved"}
    assert all(r["created_at"] == "2024-01-01T00:00:00+00:00" for r in rows)
    assert len({uuid.UUID(r["id"]) for r in rows}) == 2
    assert invoke(service, "/search", query)[1] == response
    assert invoke(service, "/search", {**query, "user_id": "unknown"})[1] == {"data": []}
    with sessions() as db:
        assert all(m.hit_count == 0 for m in db.query(AgentMemory))
        uid, pid = _identity("alpha")
        assert all(m.project_id == pid and m.customer_id == _NS
                   for m in db.query(AgentMemory).filter(AgentMemory.user_id == uid))


def test_retry_exactly_once_and_changed_payload_conflicts(api):
    service, sessions = api
    payload = chunk()
    assert invoke(service, "/add", payload)[0] == 200
    assert invoke(service, "/add", payload)[0] == 200
    assert invoke(service, "/add", chunk(content="changed"))[0] == 409
    assert invoke(service, "/add", {**payload, "messages": payload["messages"] * 2})[0] == 409
    with sessions() as db:
        assert db.query(AgentMemory).count() == 1
        assert db.query(MemoryAPIRequest).count() == 1
    # Same request_id in a different platform scope is independent.
    assert invoke(service, "/add", chunk(user="beta"))[0] == 200


def test_failed_chunk_rolls_back_receipt_and_all_sources(api, monkeypatch):
    service, sessions = api
    from memory.store import PostgresMemoryStore
    original = PostgresMemoryStore._populate_search_fields
    calls = []
    def broken(self, row, **kwargs):
        calls.append(row)
        if len(calls) == 1:
            original(self, row, **kwargs)
        # Missing vector on the second row fails the complete chunk.
    payload = chunk(); payload["messages"].append({"role": "assistant", "content": "reply"})
    monkeypatch.setattr(PostgresMemoryStore, "_populate_search_fields", broken)
    assert invoke(service, "/add", payload)[0] == 503
    with sessions() as db:
        assert db.query(AgentMemory).count() == db.query(MemoryAPIRequest).count() == 0
    monkeypatch.setattr(PostgresMemoryStore, "_populate_search_fields", original)
    assert invoke(service, "/add", payload)[0] == 200


def test_concurrent_retries_do_not_duplicate_records(api):
    service, sessions = api
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: invoke(service, "/add", chunk())[0], range(2)))
    assert statuses == [200, 200]
    with sessions() as db:
        assert db.query(AgentMemory).count() == db.query(MemoryAPIRequest).count() == 1


def test_unsafe_or_missing_token_rejected():
    for token in ["", "short", "x" * 32 + "\n", "é" * 32, "x" * 32 + "\x01"]:
        with pytest.raises(ValueError):
            AMLService(lambda: None, token)


def test_concurrent_distinct_requests_same_world_are_retryable(api):
    service, sessions = api
    payloads = [chunk(request="parallel-a", session="a"),
                chunk(request="parallel-b", session="b")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda payload: invoke(service, "/add", payload)[0], payloads))
    # Canonical tag creation may race across distinct requests. No partial
    # sources/receipt may survive a conflict; the exact retry must succeed.
    assert all(status in {200, 409} for status in statuses)
    for payload in payloads:
        assert invoke(service, "/add", payload)[0] == 200
    with sessions() as db:
        assert db.query(AgentMemory).count() == db.query(MemoryAPIRequest).count() == 2
