"""Snapshot durability, publication races and public API contracts, without AWS."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import text

from memory import config
from memory.db import get_session
from memory.models import AgentMemory
from memory.object_store import FileObjectStore, PublicationConflict, S3ObjectStore, open_object_store
from memory.recall import project_scope, recall_facts, store_facts_verbatim
from memory.snapshots import SnapshotSession
from memory.snapshot_pages import reset_cache
from memory.store import PostgresMemoryStore


class Embedder:
    def embed(self, value, *, intent="document"):
        # Semantic synonym match with no lexical overlap.
        result = [0.0] * 384
        result[0 if any(word in value.lower() for word in ("car", "automobile")) else 1] = 1.0
        return result


@pytest.fixture
def objects(tmp_path, monkeypatch):
    reset_cache()
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Embedder())
    monkeypatch.setattr(config, "MEMORY_EMBEDDINGS_ENABLED", True)
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "object_store")
    monkeypatch.setenv("MEMORY_OBJECT_STORE_URI", tmp_path.as_uri())
    monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "writer")
    monkeypatch.setenv("MEMORY_USER", "snapshot-author")
    return FileObjectStore(tmp_path)


def write(session, content="automobile", **kwargs):
    return PostgresMemoryStore(session).write(
        user_id=uuid.UUID(int=1), agent_type="assistant", content=content, **kwargs,
    )


def test_public_api_restart_scope_and_attribution(objects, monkeypatch):
    scope = {"project_id": uuid.uuid4(), "customer_id": uuid.uuid4()}
    with project_scope(**scope):
        fact = {"summary": "automobile", "tags": ["travel"], "occurred_at": "2026-01-01"}
        assert store_facts_verbatim("assistant", [fact])["written"] == 1
        assert store_facts_verbatim("assistant", [fact])["written"] == 0
        pointer = objects.get("latest.json")
        # New sessions have no shared engine, disk DB, or persistent local cache.
        monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "reader")
        rows = recall_facts("assistant", "car")
        assert rows[0]["summary"] == "automobile"
        assert rows[0]["by"] == "snapshot-author"
        assert rows[0]["occurred_at"] == "2026-01-01"
        assert recall_facts("assistant", "car", occurred_before=datetime(2025, 1, 1, tzinfo=UTC)) == []
        assert objects.get("latest.json") == pointer  # default update_hits is harmless
    with project_scope(project_id=uuid.uuid4(), customer_id=uuid.uuid4()):
        assert recall_facts("assistant", "car") == []


def test_semantic_candidates_not_limited_to_recent_pool(objects, monkeypatch):
    monkeypatch.setattr("memory.store._CANDIDATE_LIMIT", 1)
    with SnapshotSession(objects, writer=True) as session:
        target = write(session)
        target_id = target.id
        write(session, "unrelated recent text")
    with SnapshotSession(objects) as session:
        rows, plan = PostgresMemoryStore(session).recall_with_plan(
            user_id=uuid.UUID(int=1), agent_type="assistant", query="car", k=1,
        )
        assert rows[0].id == target_id
        assert plan[0][2] is True  # genuine vector match, not a recency fallback
        assert np.asarray(rows[0].embedding)[0] == 1


def test_offset_timestamps_and_query_bounds_preserve_instants(objects, monkeypatch):
    from memory.browse import browse_memories
    store_facts_verbatim("assistant", [{"summary": "automobile", "occurred_at": "2026-01-01T01:00:00+03:00"}])
    monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "reader")
    rows = recall_facts("assistant", "car")
    assert rows[0]["occurred_at"] == "2025-12-31"
    assert recall_facts("assistant", "car", occurred_before=datetime.fromisoformat("2026-01-01T02:00:00+05:00")) == []
    assert len(recall_facts("assistant", "car", occurred_before=datetime.fromisoformat("2026-01-01T03:00:00+05:00"))) == 1
    assert len(browse_memories(occurred_before="2026-01-01T03:00:00+05:00")) == 1


def test_stale_writer_merges_independent_records(objects):
    with SnapshotSession(objects, writer=True) as first, SnapshotSession(objects, writer=True) as stale:
        write(first, "winner")
        write(stale, "independent")
    with SnapshotSession(objects) as reader:
        assert {row.content for row in reader.query(AgentMemory)} == {"winner", "independent"}


def test_same_row_conflict_never_loses_winner(objects):
    with SnapshotSession(objects, writer=True) as session:
        memory_id = write(session).id
    with SnapshotSession(objects, writer=True) as first, SnapshotSession(objects, writer=True) as stale:
        first.get(AgentMemory, memory_id).context_summary = "winner"
        stale.get(AgentMemory, memory_id).context_summary = "conflicting"
        first.commit()
        with pytest.raises(PublicationConflict):
            stale.commit()
        with pytest.raises(RuntimeError, match="failed"):
            stale.commit()
    with SnapshotSession(objects) as reader:
        assert reader.get(AgentMemory, memory_id).context_summary == "winner"


def test_reader_is_pinned_and_enforces_read_only(objects):
    with SnapshotSession(objects, writer=True) as writer:
        write(writer, "before")
    with SnapshotSession(objects) as reader:
        with SnapshotSession(objects, writer=True) as writer:
            write(writer, "after")
        assert reader.query(AgentMemory).count() == 1
        with pytest.raises(Exception, match="readonly"):
            reader.execute(text("DELETE FROM agent_memory"))
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 2
        reader.add(AgentMemory(content="blocked"))
        with pytest.raises(RuntimeError, match="read-only"):
            reader.commit()


@pytest.mark.parametrize("stage", ["snapshot", "pointer"])
def test_failure_does_not_publish_partial_state(objects, monkeypatch, stage):
    with SnapshotSession(objects, writer=True) as session:
        write(session, "durable")
    old = objects.get("latest.json")
    put = objects.put

    def fail(key, data, **kwargs):
        if (key == "latest.json") == (stage == "pointer"):
            raise OSError("simulated outage")
        return put(key, data, **kwargs)

    monkeypatch.setattr(objects, "put", fail)
    with SnapshotSession(objects, writer=True) as session:
        with pytest.raises(OSError):
            write(session, "unpublished")
    assert objects.get("latest.json") == old
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1


def test_lost_publication_response_can_retry_idempotently(objects, monkeypatch):
    put = FileObjectStore.put

    def lose_response(self, key, data, **kwargs):
        result = put(self, key, data, **kwargs)
        if key == "latest.json":
            raise OSError("response lost after durable publication")
        return result

    monkeypatch.setattr(FileObjectStore, "put", lose_response)
    with pytest.raises(OSError):
        store_facts_verbatim("assistant", ["durable fact"])
    monkeypatch.setattr(FileObjectStore, "put", put)
    assert store_facts_verbatim("assistant", ["durable fact"])["written"] == 0
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1


def test_transaction_context_also_publishes(objects):
    with SnapshotSession(objects, writer=True) as session:
        with session.begin():
            session.add(AgentMemory(user_id=uuid.UUID(int=1), agent_type="assistant", content="context"))
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).one().content == "context"


def test_nested_commit_never_publishes_before_outer_commit(objects):
    with SnapshotSession(objects, writer=True) as session:
        write(session, "original")
    old = objects.get("latest.json")
    with SnapshotSession(objects, writer=True) as session:
        session.add(AgentMemory(user_id=uuid.UUID(int=1), agent_type="assistant", content="outer"))
        session.flush()
        with session.begin_nested():
            session.add(AgentMemory(user_id=uuid.UUID(int=1), agent_type="assistant", content="nested"))
        assert objects.get("latest.json") == old
        session.rollback()
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).one().content == "original"


def test_snapshot_recovery_in_separate_process(objects):
    with SnapshotSession(objects, writer=True) as writer:
        write(writer, "separate process recovery")
    code = """
from memory.models import AgentMemory
from memory.object_store import open_object_store
from memory.snapshots import SnapshotSession
import sys
with SnapshotSession(open_object_store(sys.argv[1])) as reader:
    assert reader.query(AgentMemory).one().content == 'separate process recovery'
"""
    subprocess.run([sys.executable, "-c", code, objects.root.as_uri()], check=True, capture_output=True)


def test_first_write_savepoint_is_rolled_back_with_outer_transaction(objects):
    with SnapshotSession(objects, writer=True) as session:
        session.commit()
    old = objects.get("latest.json")
    with SnapshotSession(objects, writer=True) as session:
        with pytest.raises(ValueError, match="abort outer"):
            with session.begin():
                with session.begin_nested():
                    session.add(AgentMemory(user_id=uuid.UUID(int=1), agent_type="assistant",
                                            content="aborted nested write"))
                raise ValueError("abort outer")
        assert session.query(AgentMemory).count() == 0
        assert objects.get("latest.json") == old
        write(session, "successful unrelated commit")
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).one().content == "successful unrelated commit"


def test_source_retry_and_embedding_failure(objects, monkeypatch):
    source = uuid.uuid4()
    with SnapshotSession(objects, writer=True) as session:
        original = write(session, source_message_id=source)
        original_id = original.id
    with SnapshotSession(objects, writer=True) as session:
        assert write(session, source_message_id=source).id == original_id
        assert session.query(AgentMemory).count() == 1
    class Unavailable:
        def embed(self, *args, **kwargs):
            return None
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _: Unavailable())
    with SnapshotSession(objects, writer=True) as session:
        with pytest.raises(RuntimeError, match="requires embeddings"):
            write(session, "cannot embed")
    with SnapshotSession(objects) as session:
        assert session.query(AgentMemory).count() == 1


def test_merge_history_survives_restart(objects):
    from memory.browse import browse_memories
    from memory.mutate import merge_memories

    store_facts_verbatim("assistant", ["first record", "second record"])
    rows = browse_memories()
    merge_memories([row["id"] for row in rows])
    with SnapshotSession(objects) as reader:
        all_rows = reader.query(AgentMemory).all()
        assert len(all_rows) == 3
        children = [row for row in all_rows if row.parent_id is not None]
        assert len(children) == 2
        assert all(not row.is_active and row.valid_to and row.superseded_by_id for row in children)
        assert {row.content for row in children} == {"first record", "second record"}


@pytest.mark.parametrize("damage", ["corrupt", "missing", "profile", "format"])
def test_invalid_publication_fails_closed(objects, damage):
    with SnapshotSession(objects, writer=True) as session:
        write(session)
    pointer = objects.get("latest.json")
    manifest = json.loads(pointer.data)
    index = json.loads(objects.get(f"manifests/{manifest['sha256']}.json").data)
    path = objects.root / "chunks" / index["chunks"][0]
    if damage == "corrupt":
        path.write_bytes(b"corrupt")
    elif damage == "missing":
        path.unlink()
    else:
        manifest["embedding" if damage == "profile" else "format"] = "unsupported"
        objects.put("latest.json", json.dumps(manifest).encode(), expected=pointer.etag)
    reset_cache()
    with pytest.raises(ValueError):
        SnapshotSession(objects)


def test_missing_reader_and_invalid_config_fail_clearly(objects, monkeypatch):
    with pytest.raises(RuntimeError, match="initialize"):
        SnapshotSession(objects)
    monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "both")
    with pytest.raises(ValueError, match="ROLE"):
        get_session()
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "typo")
    with pytest.raises(ValueError, match="BACKEND"):
        get_session()
    with pytest.raises(ValueError):
        open_object_store("https://example.com")
    with pytest.raises(ValueError):
        objects.get("../escape")


def test_http_atomic_ingestion_retries_and_read_replica(objects, monkeypatch):
    from memory.aml_api import AMLService
    from memory.models import MemoryAPIRequest
    from test_aml_api import TOKEN, chunk, invoke

    service = AMLService(get_session, TOKEN)
    request = chunk()
    request["messages"].append({"role": "assistant", "content": "automobile"})
    assert invoke(service, "/add", request)[0] == 200
    assert invoke(service, "/add", request)[0] == 200
    assert invoke(service, "/add", chunk(content="different"))[0] == 409
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 2
        assert reader.query(MemoryAPIRequest).count() == 1
    monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "reader")
    assert invoke(service, "/add", chunk(request="next"))[0] == 403
    assert invoke(service, "/search", {"user_id": "alpha", "query": "car", "top_k": 5})[0] == 200
    assert invoke(service, "/health", auth=False, method="GET")[0] == 200


def test_http_failed_chunk_keeps_receipt_and_sources_out_of_snapshot(objects, monkeypatch):
    from memory.aml_api import AMLService
    from memory.models import MemoryAPIRequest
    from test_aml_api import TOKEN, chunk, invoke

    with SnapshotSession(objects, writer=True) as writer:
        writer.commit()
    class FailSecond:
        calls = 0
        def embed(self, value, **kwargs):
            self.calls += 1
            return Embedder().embed(value) if self.calls == 1 else None
    embedder = FailSecond()
    monkeypatch.setattr("memory.store.get_embedding_service", lambda _: embedder)
    request = chunk()
    request["messages"].append({"role": "assistant", "content": "second"})
    assert invoke(AMLService(get_session, TOKEN), "/add", request)[0] == 503
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 0
        assert reader.query(MemoryAPIRequest).count() == 0


def test_http_publication_conflict_is_retryable(objects, monkeypatch):
    from memory.aml_api import AMLService
    from test_aml_api import TOKEN, chunk, invoke

    put = FileObjectStore.put
    def conflict(self, key, data, **kwargs):
        if key == "latest.json":
            raise PublicationConflict("stale writer")
        return put(self, key, data, **kwargs)
    monkeypatch.setattr(FileObjectStore, "put", conflict)
    assert invoke(AMLService(get_session, TOKEN), "/add", chunk())[0] == 409


def test_http_snapshot_server_startup(objects, monkeypatch):
    from memory import aml_api
    from test_aml_api import TOKEN

    class LocalModel:
        def warm_load(self):
            return True
    class Server:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def serve_forever(self):
            pass
    monkeypatch.setenv("MEMORY_API_TOKEN", TOKEN)
    monkeypatch.setattr(aml_api, "get_embedding_service", lambda _: LocalModel())
    monkeypatch.setattr(aml_api, "make_server", lambda *args, **kwargs: Server())
    aml_api.main([])
    assert objects.get("latest.json") is not None
    monkeypatch.setenv("MEMORY_OBJECT_STORE_ROLE", "reader")
    aml_api.main([])


def test_s3_wire_protocol_and_error_propagation():
    boto3 = pytest.importorskip("boto3")
    from botocore.exceptions import ClientError
    from botocore.stub import Stubber

    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    store = S3ObjectStore("bucket", "corpus", client=client)
    with Stubber(client) as stub:
        stub.add_response("put_object", {"ETag": '"one"'},
                          {"Bucket": "bucket", "Key": "corpus/latest.json", "Body": b"1", "IfNoneMatch": "*"})
        assert store.put("latest.json", b"1") == '"one"'
        stub.add_response("put_object", {"ETag": '"two"'},
                          {"Bucket": "bucket", "Key": "corpus/latest.json", "Body": b"2", "IfMatch": '"one"'})
        assert store.put("latest.json", b"2", expected='"one"') == '"two"'
        stub.add_client_error("put_object", "PreconditionFailed", http_status_code=412)
        with pytest.raises(PublicationConflict):
            store.put("latest.json", b"stale", expected='"one"')
        stub.add_client_error("get_object", "NoSuchKey", http_status_code=404)
        assert store.get("missing") is None
        stub.add_client_error("get_object", "AccessDenied", http_status_code=403)
        with pytest.raises(ClientError):
            store.get("private")
        body = io.BytesIO(b"db")
        stub.add_response("get_object", {"Body": body, "ETag": '"two"', "ContentLength": 2})
        assert store.get("latest.json").data == b"db"
        assert body.closed
        oversized = io.BytesIO(b"x" * 100)
        stub.add_response("get_object", {"Body": oversized, "ETag": '"large"', "ContentLength": 100})
        with pytest.raises(ValueError, match="MAX_OBJECT_BYTES"):
            store.get("chunk", max_bytes=10)
        assert oversized.closed
        stub.assert_no_pending_responses()
