"""Disposable SQLite sessions, chunked snapshots and optimistic publication.

There is no authoritative local database. Each session loads exactly one published
version into memory. Writers upload changed chunks and conditionally publish a
manifest, rebasing independent concurrent row changes. Failed sessions are closed.
"""
from __future__ import annotations

import os
import uuid

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from memory import config
from memory.models import Base, MemoryTag
from memory.object_store import ObjectStore, PublicationConflict, open_object_store
from memory.snapshot_pages import load, publish
from memory.snapshot_merge import merge


def _profile() -> dict:
    return {"backend": config.MEMORY_EMBEDDING_BACKEND,
            "model": config.EMBEDDING_MODEL, "dimensions": config.EMBEDDING_DIM}


def _sqlite_connect(connection, record):
    # Let SQLAlchemy control BEGIN. Legacy sqlite3 mode otherwise treats a first
    # SAVEPOINT as the outer transaction and RELEASE survives an outer rollback.
    connection.isolation_level = None
    connection.execute("PRAGMA foreign_keys=ON")


def _sqlite_begin(connection):
    connection.exec_driver_sql("BEGIN")


class SnapshotSession(Session):
    def __init__(self, objects: ObjectStore, *, writer: bool = False):
        if config.EMBEDDING_DIM != 384:
            raise ValueError("Snapshot backend requires 384-dimensional embeddings")
        self.objects = objects
        self.writer = writer
        self.failed = False
        self.etag = None
        self.snapshot_engine = create_engine("sqlite://", poolclass=StaticPool)
        event.listen(self.snapshot_engine, "connect", _sqlite_connect)
        event.listen(self.snapshot_engine, "begin", _sqlite_begin)
        try:
            data, self.etag, self.known_chunks = load(objects, _profile())
            with self.snapshot_engine.connect() as conn:
                raw = conn.connection.driver_connection
                if data is not None:
                    raw.deserialize(data)
                    if raw.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise ValueError("Invalid SQLite snapshot")
                elif not writer:
                    raise RuntimeError("No published snapshot; initialize with a writer first")
                raw.execute("PRAGMA foreign_keys=ON")
            if data is None:
                Base.metadata.create_all(self.snapshot_engine)
                with self.snapshot_engine.connect() as conn:
                    data = conn.connection.driver_connection.serialize()
            self.base_data = data if writer else None
            if not writer:
                with self.snapshot_engine.connect() as conn:
                    conn.exec_driver_sql("PRAGMA query_only=ON")
            super().__init__(bind=self.snapshot_engine, expire_on_commit=True,
                             info={"snapshot_backend": True, "snapshot_reader": not writer})
            # Cover both session.commit() and `with session.begin():` so a
            # transaction cannot acknowledge a local-only write by accident.
            event.listen(self, "before_commit", lambda session: self._check())
            event.listen(self, "after_commit", self._publish)
            event.listen(self, "before_flush", self._canonical_tags)
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

    def _canonical_tags(self, session, context, instances):
        for row in session.new:
            if isinstance(row, MemoryTag):
                row.id = uuid.uuid5(uuid.NAMESPACE_URL, f"mempp-tag:{row.scope_key}:{row.label}")

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
            ours = data
            retries = int(os.environ.get("MEMORY_OBJECT_STORE_WRITE_RETRIES", "8"))
            if retries < 1:
                raise ValueError("MEMORY_OBJECT_STORE_WRITE_RETRIES must be positive")
            for attempt in range(retries):
                try:
                    self.etag = publish(self.objects, data, _profile(), self.known_chunks, self.etag)
                    break
                except PublicationConflict:
                    if attempt + 1 == retries:
                        raise
                    latest, self.etag, known = load(self.objects, _profile())
                    if latest is None:
                        raise PublicationConflict("Published corpus disappeared; reload and retry")
                    self.known_chunks.update(known)
                    data = merge(self.base_data, ours, latest)
            if data != ours:
                with self.snapshot_engine.connect() as conn:
                    conn.connection.driver_connection.deserialize(data)
                    conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            self.base_data = data
        except BaseException:
            self.failed = True
            raise

    def close(self):
        try:
            super().close()
        finally:
            self.failed = True
            self.base_data = None
            self.snapshot_engine.dispose()


def get_snapshot_session() -> SnapshotSession:
    role = os.environ.get("MEMORY_OBJECT_STORE_ROLE", "reader")
    if role not in {"reader", "writer"}:
        raise ValueError("MEMORY_OBJECT_STORE_ROLE must be reader or writer")
    uri = os.environ.get("MEMORY_OBJECT_STORE_URI", "")
    return SnapshotSession(open_object_store(uri), writer=role == "writer")
