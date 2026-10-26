"""Optimistic row-level merge for concurrently published SQLite snapshots.

Independent inserts/edits merge; incompatible edits to the same row are rejected.
This is not serializable predicate locking or last-write-wins reconciliation.
"""
from __future__ import annotations

import sqlite3

from memory.models import Base
from memory.object_store import PublicationConflict


def _connection(data):
    connection = sqlite3.connect(":memory:")
    connection.deserialize(data)
    return connection


def _same_retry(table, ours, theirs):
    if ours is None or theirs is None:
        return False
    if table == "memory_api_request":
        return ours["payload_sha256"] == theirs["payload_sha256"]
    if table == "agent_memory" and ours["source_message_id"] is not None:
        ignore = {"created_at", "last_used_at"}
        if ours["occurred_at"] is None and theirs["occurred_at"] is None:
            ignore.add("valid_from")
        return {k: v for k, v in ours.items() if k not in ignore} == {
            k: v for k, v in theirs.items() if k not in ignore}
    return False


def merge(base_data, ours_data, latest_data):
    connections = [_connection(data) for data in (base_data, ours_data, latest_data)]
    base, ours, latest = connections
    try:
        latest.execute("PRAGMA foreign_keys=ON")
        latest.execute("BEGIN")
        latest.execute("PRAGMA defer_foreign_keys=ON")
        for table in Base.metadata.sorted_tables:
            name = table.name
            keys = [column.name for column in table.primary_key]
            def rows(connection):
                cursor = connection.execute(f'SELECT * FROM "{name}"')
                names = [col[0] for col in cursor.description]
                result = [dict(zip(names, values)) for values in cursor]
                return {tuple(row[key] for key in keys): row for row in result}
            original, changed, current = [rows(conn) for conn in connections]
            for key in original.keys() | changed.keys():
                before, after, now = original.get(key), changed.get(key), current.get(key)
                if before == after:
                    continue
                if now != before:
                    if after is None and now is None:
                        continue
                    if before is None and (now == after or _same_retry(name, after, now)):
                        continue
                    raise PublicationConflict(f"Concurrent changes to {name}; reload and retry")
                where = " AND ".join(f'"{column}"=?' for column in keys)
                if after is None:
                    latest.execute(f'DELETE FROM "{name}" WHERE {where}', key)
                elif before is None:
                    columns = list(after)
                    fields = ",".join(f'"{column}"' for column in columns)
                    marks = ",".join("?" for _ in columns)
                    latest.execute(f'INSERT INTO "{name}" ({fields}) VALUES ({marks})', list(after.values()))
                else:
                    updates = ",".join(f'"{column}"=?' for column in after)
                    latest.execute(f'UPDATE "{name}" SET {updates} WHERE {where}', [*after.values(), *key])
        latest.commit()
        if latest.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise PublicationConflict("Concurrent changes would break provenance references")
        return latest.serialize()
    except sqlite3.IntegrityError as exc:
        raise PublicationConflict("Concurrent changes violate a record or tag constraint") from exc
    finally:
        for connection in connections:
            connection.close()
