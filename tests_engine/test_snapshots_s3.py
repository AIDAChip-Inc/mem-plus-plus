"""Opt-in S3 contract test; creates/deletes ONLY its own random object prefix."""
import os
import uuid

import pytest


pytestmark = pytest.mark.skipif(not os.environ.get("MEMORY_TEST_S3_URI"), reason="Set MEMORY_TEST_S3_URI for live S3 validation")


@pytest.fixture
def live_store():
    pytest.importorskip("boto3")
    from memory.object_store import open_object_store
    root = os.environ["MEMORY_TEST_S3_URI"].rstrip("/")
    assert root.startswith("s3://")
    store = open_object_store(root + "/mempp-test-" + uuid.uuid4().hex)
    try:
        yield store
    finally:
        # Do not list or delete the caller's other data. No bucket creation here.
        paginator = store.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=store.bucket, Prefix=store.prefix + "/"):
            for obj in page.get("Contents", []):
                store.client.delete_object(Bucket=store.bucket, Key=obj["Key"])


def test_live_s3_publication_restart_and_conflict(live_store, monkeypatch):
    from memory.models import AgentMemory
    from memory.object_store import PublicationConflict
    from memory.snapshots import SnapshotSession
    from test_snapshots import Embedder, write

    store = live_store
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Embedder())
    with SnapshotSession(store, writer=True) as first, SnapshotSession(store, writer=True) as stale:
        write(first, "durable original")
        with pytest.raises(PublicationConflict):
            write(stale, "rejected stale write")
    with SnapshotSession(store) as reader:
        assert reader.query(AgentMemory).one().content == "durable original"
        with SnapshotSession(store, writer=True) as writer:
            write(writer, "next published version")
        assert reader.query(AgentMemory).count() == 1
    with SnapshotSession(store) as reader:
        assert reader.query(AgentMemory).count() == 2


def test_live_s3_with_real_embeddings_and_atomic_http_receipt(live_store):
    from datetime import datetime
    from memory.aml_api import AMLService, AddRequest, SearchRequest
    from memory.embeddings import get_embedding_service
    from memory.models import AgentMemory, MemoryAPIRequest
    from memory.snapshots import SnapshotSession

    if not get_embedding_service("memory").warm_load():
        pytest.skip("Real embedding model unavailable")
    token = "s3-test-only-token-with-no-real-secret-0123456789"
    writer = AMLService(lambda: SnapshotSession(live_store, writer=True), token)
    timestamp = int(datetime.fromisoformat("2026-01-01T03:00:00+03:00").timestamp() * 1000)
    request = AddRequest(request_id="retry-safe", user_id="team-a", session_id="session-a", messages=[
        {"role": "user", "content": "The automobile is parked in the garage.", "timestamp": timestamp},
        {"role": "assistant", "content": "The finance meeting is on Tuesday."},
    ])
    assert writer.add(request)["success"]
    assert writer.add(request)["success"]
    with SnapshotSession(live_store) as reader:
        assert reader.query(AgentMemory).count() == 2
        assert reader.query(MemoryAPIRequest).count() == 1
    reader = AMLService(lambda: SnapshotSession(live_store), token)
    result = reader.search(SearchRequest(user_id="team-a", query="Where is the car?", top_k=2))
    row = next(row for row in result["data"] if "automobile" in row["content"])
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"
    assert reader.search(SearchRequest(user_id="team-b", query="Where is the car?", top_k=2)) == {"data": []}
