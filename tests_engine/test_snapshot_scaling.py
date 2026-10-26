"""Concurrent writers, transfer volume, disposable caches and format upgrades."""
import json
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func

from memory.aml_api import AMLService, AddRequest
from memory.models import AgentMemory, MemoryTag
from memory.object_store import PublicationConflict
from memory.snapshot_pages import digest, reset_cache
from memory.snapshots import SnapshotSession, _profile
from test_snapshots import objects as objects, write


def test_parallel_writers_preserve_records_and_canonical_tags(objects):
    barrier = threading.Barrier(5)
    def ingest(index):
        with SnapshotSession(objects, writer=True) as session:
            barrier.wait(timeout=15)
            write(session, f"record {index}", tags=["shared"])
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(ingest, range(5)))
    with SnapshotSession(objects) as reader:
        assert {row.content for row in reader.query(AgentMemory)} == {f"record {i}" for i in range(5)}
        assert reader.query(MemoryTag).count() == 1
        assert all(len(row.tags) == 1 for row in reader.query(AgentMemory))


def test_concurrent_identical_source_is_idempotent(objects):
    source = uuid.uuid4()
    with SnapshotSession(objects, writer=True) as first, SnapshotSession(objects, writer=True) as other:
        one = write(first, source_message_id=source)
        two = write(other, source_message_id=source)
        assert one.id == two.id
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1


def test_concurrent_identical_http_receipt_is_idempotent(objects):
    barrier = threading.Barrier(2)
    def sessions():
        session = SnapshotSession(objects, writer=True)
        barrier.wait(timeout=15)
        return session
    request = AddRequest(request_id="same", user_id="same", session_id="same", messages=[
        {"role": "user", "content": "automobile"},
    ])
    service = AMLService(sessions, "concurrent-test-token-no-real-secret-0123456789")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service.add(request), range(2)))
    assert all(result["success"] for result in results)
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1


def test_concurrent_different_request_payload_is_rejected(objects):
    from memory.models import MemoryAPIRequest
    request_id = uuid.uuid4()
    with SnapshotSession(objects, writer=True) as first, SnapshotSession(objects, writer=True) as other:
        first.add(MemoryAPIRequest(id=request_id, payload_sha256="a" * 64))
        other.add(MemoryAPIRequest(id=request_id, payload_sha256="b" * 64))
        first.commit()
        with pytest.raises(PublicationConflict):
            other.commit()


def test_same_value_counter_changes_are_not_mistaken_for_duplicate_inserts(objects):
    with SnapshotSession(objects, writer=True) as writer:
        memory_id = write(writer).id
    with SnapshotSession(objects, writer=True) as first, SnapshotSession(objects, writer=True) as other:
        first.get(AgentMemory, memory_id).hit_count += 1
        other.get(AgentMemory, memory_id).hit_count += 1
        first.commit()
        with pytest.raises(PublicationConflict):
            other.commit()
    with SnapshotSession(objects, writer=True) as retry:
        retry.get(AgentMemory, memory_id).hit_count += 1
        retry.commit()
    with SnapshotSession(objects) as reader:
        assert reader.get(AgentMemory, memory_id).hit_count == 2


def test_chunk_reuse_and_warm_reader_transfer_volume(objects, monkeypatch):
    # A sizable immutable payload fills many chunks. A small independent write
    # should upload only changed chunks, never the entire corpus again.
    with SnapshotSession(objects, writer=True) as writer:
        writer.execute(AgentMemory.__table__.insert().values(
            user_id=uuid.UUID(int=1), agent_type="assistant", content=func.zeroblob(4 * 1024**2)))
        writer.commit()
    uploaded = []
    downloaded = []
    put, get = objects.put, objects.get
    def counted_put(key, data, **kwargs):
        uploaded.append((key, len(data)))
        return put(key, data, **kwargs)
    def counted_get(key):
        downloaded.append(key)
        return get(key)
    monkeypatch.setattr(objects, "put", counted_put)
    monkeypatch.setattr(objects, "get", counted_get)
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1
    assert downloaded == ["latest.json"]
    with SnapshotSession(objects, writer=True) as writer:
        write(writer, "small incremental addition")
    assert sum(size for key, size in uploaded) < 1024**2
    assert not any(key.startswith("snapshots/") for key, size in uploaded)
    downloaded.clear()
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 2
    assert downloaded == ["latest.json"]


def test_disk_cache_survives_process_cache_loss_and_corruption(objects, monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_OBJECT_STORE_CACHE_DIR", str(tmp_path / "cache"))
    with SnapshotSession(objects, writer=True) as writer:
        write(writer)
    reset_cache()
    get = objects.get
    def pointer_only(key):
        assert key == "latest.json", "unchanged immutable objects should come from disk cache"
        return get(key)
    monkeypatch.setattr(objects, "get", pointer_only)
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1
    code = """
from memory.models import AgentMemory
from memory.object_store import FileObjectStore, open_object_store
from memory.snapshots import SnapshotSession
import sys
get = FileObjectStore.get
def pointer_only(self, key):
    assert key == 'latest.json', key
    return get(self, key)
FileObjectStore.get = pointer_only
with SnapshotSession(open_object_store(sys.argv[1])) as reader:
    assert reader.query(AgentMemory).count() == 1
"""
    subprocess.run([sys.executable, "-c", code, objects.root.as_uri()], check=True, capture_output=True)
    monkeypatch.setattr(objects, "get", get)
    reset_cache()
    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            path.write_bytes(b"corrupt disposable cache")
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1


def test_legacy_snapshot_read_and_upgrade(objects):
    with SnapshotSession(objects, writer=True) as writer:
        write(writer)
        data = writer.base_data
    sha = digest(data)
    objects.put(f"snapshots/{sha}.sqlite", data)
    pointer = objects.get("latest.json")
    objects.put("latest.json", json.dumps({"format": 1, "embedding": _profile(), "sha256": sha}).encode(), expected=pointer.etag)
    reset_cache()
    with SnapshotSession(objects, writer=True) as writer:
        assert writer.query(AgentMemory).count() == 1
        write(writer, "upgraded")
    assert json.loads(objects.get("latest.json").data)["format"] == 2
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 2


def test_operator_size_limit_is_optional_and_enforced(objects, monkeypatch):
    monkeypatch.setenv("MEMORY_OBJECT_STORE_MAX_BYTES", "1")
    with SnapshotSession(objects, writer=True) as writer:
        with pytest.raises(ValueError, match="MAX_BYTES"):
            write(writer)
    assert objects.get("latest.json") is None
    monkeypatch.setenv("MEMORY_OBJECT_STORE_MAX_BYTES", "0")
    with SnapshotSession(objects, writer=True) as writer:
        write(writer)


def test_rebase_respects_foreign_keys(objects):
    with SnapshotSession(objects, writer=True) as writer:
        memory_id = write(writer).id
    with SnapshotSession(objects, writer=True) as deletion, SnapshotSession(objects, writer=True) as linking:
        row = linking.get(AgentMemory, memory_id)
        # Add a referencing row without changing its parent. If a concurrent
        # delete wins, rebasing must reject the dangling provenance reference.
        linking.add(AgentMemory(user_id=row.user_id, agent_type=row.agent_type,
                               content="child", parent_id=memory_id))
        deletion.delete(deletion.get(AgentMemory, memory_id))
        deletion.commit()
        with pytest.raises(PublicationConflict):
            linking.commit()
