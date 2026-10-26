"""Disposable SQLite sessions, durable immutable snapshots and CAS publication.

There is no authoritative local database. Each session loads exactly one published
version into memory. A successful writer commit uploads the complete DB before
conditionally publishing its manifest. A failed publication poisons the session.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from memory import config
from memory.models import Base
from memory.object_store import MAX_SNAPSHOT_BYTES, ObjectStore, PublicationConflict, open_object_store


def _profile() -> dict:
    return {"backend": config.MEMORY_EMBEDDING_BACKEND,
            "model": config.EMBEDDING_MODEL, "dimensions": config.EMBEDDING_DIM}


class SnapshotSession(Session):
    def __init__(self, objects: ObjectStore, *, writer: bool = False):
        if config.EMBEDDING_DIM != 384:
            raise ValueError("Snapshot format v1 requires 384-dimensional embeddings")
        self.objects = objects
        self.writer = writer
        self.failed = False
        self.etag = None
        self.snapshot_engine = create_engine("sqlite://", poolclass=StaticPool)
        try:
            pointer = objects.get("latest.json")
            with self.snapshot_engine.connect() as conn:
                raw = conn.connection.driver_connection
                if pointer is not None:
                    manifest = json.loads(pointer.data)
                    digest = manifest.get("sha256", "")
                    if (manifest.get("format") != 1
                            or not re.fullmatch(r"[0-9a-f]{64}", digest)
                            or manifest.get("embedding") != _profile()):
                        raise ValueError("Unsupported snapshot format or embedding profile")
                    snapshot = objects.get(f"snapshots/{digest}.sqlite")
                    if snapshot is None or hashlib.sha256(snapshot.data).hexdigest() != digest:
                        raise ValueError("Published snapshot is missing or has an invalid checksum")
                    if len(snapshot.data) > MAX_SNAPSHOT_BYTES:
                        raise ValueError("Snapshot exceeds the 256 MiB limit")
                    raw.deserialize(snapshot.data)
                    if raw.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise ValueError("Invalid SQLite snapshot")
                    self.etag = pointer.etag
                elif not writer:
                    raise RuntimeError("No published snapshot; initialize with a writer first")
            if pointer is None:
                Base.metadata.create_all(self.snapshot_engine)
            if not writer:
                with self.snapshot_engine.connect() as conn:
                    conn.exec_driver_sql("PRAGMA query_only=ON")
            super().__init__(bind=self.snapshot_engine, expire_on_commit=True,
                             info={"snapshot_backend": True, "snapshot_reader": not writer})
            # Cover both session.commit() and `with session.begin():` so a
            # transaction cannot acknowledge a local-only write by accident.
            event.listen(self, "before_commit", lambda session: self._check())
            event.listen(self, "after_commit", self._publish)
        except BaseException:
            self.snapshot_engine.dispose()
            raise

    def _check(self):
        if self.failed:
            raise RuntimeError("Snapshot session failed; close it and retry in a fresh session")

    def execute(self, *args, **kwargs):
        self._check()
        return super().execute(*args, **kwargs)

    def flush(self, *args, **kwargs):
        self._check()
        if not self.writer and (self.new or self.dirty or self.deleted):
            raise RuntimeError("Snapshot reader is read-only; use the ingestion writer")
        return super().flush(*args, **kwargs)

    def commit(self):
        self._check()
        return super().commit()

    def _publish(self, session):
        self._check()
        if session.in_nested_transaction():
            return  # SAVEPOINT release is not a durable outer commit.
        # Only writer commits publish. Readers never persist query statistics.
        try:
            if not self.writer:
                return
            with self.snapshot_engine.connect() as conn:
                data = conn.connection.driver_connection.serialize()
            if len(data) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot exceeds the 256 MiB limit")
            digest = hashlib.sha256(data).hexdigest()
            key = f"snapshots/{digest}.sqlite"
            try:
                self.objects.put(key, data)
            except PublicationConflict:
                # An identical snapshot can already exist after a lost response.
                existing = self.objects.get(key)
                if existing is None or existing.data != data:
                    raise ValueError("Immutable snapshot key contains different data")
            manifest = json.dumps({"format": 1, "sha256": digest, "embedding": _profile()},
                                  sort_keys=True).encode()
            self.etag = self.objects.put("latest.json", manifest, expected=self.etag)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        try:
            super().close()
        finally:
            self.failed = True
            self.snapshot_engine.dispose()


def get_snapshot_session() -> SnapshotSession:
    role = os.environ.get("MEMORY_OBJECT_STORE_ROLE", "reader")
    if role not in {"reader", "writer"}:
        raise ValueError("MEMORY_OBJECT_STORE_ROLE must be reader or writer")
    uri = os.environ.get("MEMORY_OBJECT_STORE_URI", "")
    return SnapshotSession(open_object_store(uri), writer=role == "writer")
