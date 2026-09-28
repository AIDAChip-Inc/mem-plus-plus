"""User-invokable memory WRITE ops — supersede, merge, reset.

The original engine port deliberately left the supersession/consolidation WRITE
path out of scope (recall + write core only). This module adds the three
**manual, user-invokable** mutation ops the Memory-tab UI drives (Zain builds the
action menu; this is the engine it calls):

* ``supersede_memory`` — retire one memory (bi-temporal soft supersession).
* ``merge_memories``   — consolidate >=2 memories into one new parent.
* ``reset_db``         — empty the three memory tables (the UI "Reset" button).

Faithfulness to production (``app/services/memory/store.py`` +
``app/services/memory/consolidation.py``) and the DELIBERATE differences a manual
user action must make vs production's AUTOMATIC consolidation:

1. **Supersede — ``is_active`` differs by design.** Production's automatic
   value-conflict supersession (``_supersede_conflict``) sets ``valid_to`` +
   ``superseded_by_id`` but KEEPS ``is_active=True`` — the "queryable-A" property,
   so an auto-superseded row stays recallable. A MANUAL user supersede is a
   stronger, explicit RETIRE: the user is intentionally removing the fact from
   recall, so we ALSO set ``is_active=False`` (recall filters ``is_active`` — see
   ``store._recall_scoped``), which is exactly what makes "recall no longer
   returns it" true. Still a SOFT delete (no DELETE): the row, its ``valid_to``,
   and its ``superseded_by_id`` audit pointer all persist.

2. **Merge — deterministic summary, NO LLM.** Production's union-merge
   (``_insert_root_archive_sources``) synthesizes the parent text with a Haiku
   call (``consolidation.MERGE_PROMPT``). A manual merge must be storage-only and
   offline-safe, so when no ``summary`` is supplied we build a DETERMINISTIC union
   of the children's distinct summaries instead (no LLM dependency). Everything
   else mirrors production's root exactly: identity inherited from the newest
   child, ``created_at``/``occurred_at`` backdated to the earliest child,
   ``valid_from = earliest_occ or earliest_created``, ``hit_count=0``,
   ``parent_id=None``, union-of-tags provenance, and the parent re-embedded via
   the SAME ``store._populate_search_fields`` path ``write`` uses.

3. **Merge — children get the FULL bi-temporal audit trail.** Production's
   union-merge archive sets only ``is_active=False`` + ``parent_id=root`` on the
   children; its conflict path sets only ``valid_to`` + ``superseded_by_id``. A
   manual merge records BOTH — ``is_active=False``, ``parent_id=parent``,
   ``valid_to=now``, ``superseded_by_id=parent`` — so a user-driven merge leaves a
   complete, auditable "these N were merged INTO that parent" trail.

4. **Reset uses DELETE, not TRUNCATE.** DELETE is dialect-agnostic (runs on the
   SQLite-degraded path too), returns row counts, and is idempotent — functionally
   the same in-DB wipe. This is the SOFT in-DB reset; the CLI ``run_demo.sh reset``
   is the harder full-cluster wipe.

5. **``reason`` is accepted but not persisted.** Production's schema has no
   supersession-reason column, so persisting one would diverge from the faithful
   model (and need a migration). ``reason`` is accepted for API/UI parity and
   logged; it is not stored.

Scope: every op operates over the SAME project pool the UI browses — the
``(project_id, customer_id)`` derived by ``recall._scope`` (see ``browse.py``). An
id that does not exist, is malformed, or is out of that scope raises a clear
``ValueError``.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from memory.consolidation import deterministic_union_summary, insert_root_archive_sources
from memory.db import get_session
from memory.models import AgentMemory, MemoryTag, MemoryTagLink
from memory.recall import _scope

logger = logging.getLogger(__name__)


def _as_uuid(memory_id) -> uuid.UUID:
    """Coerce an incoming id (str or UUID) to a UUID; clear error otherwise."""
    if isinstance(memory_id, uuid.UUID):
        return memory_id
    try:
        return uuid.UUID(str(memory_id))
    except (ValueError, TypeError):
        raise ValueError(f"invalid memory id: {memory_id!r}") from None


def _get_in_scope(db, memory_id) -> AgentMemory:
    """Load a memory by id and assert it is in the UI's project-pool scope.

    In-scope == the ``(project_id, customer_id)`` pool ``recall._scope`` derives —
    the exact pool ``browse_memories`` enumerates. Persona (``agent_type``) is NOT
    part of the gate: the UI can act on any persona's row in the project pool.
    """
    mid = _as_uuid(memory_id)
    _uid, pid, cid, _slug = _scope()
    m = db.get(AgentMemory, mid)
    if m is None:
        raise ValueError(f"memory {memory_id} not found")
    if m.project_id != pid or m.customer_id != cid:
        raise ValueError(f"memory {memory_id} is out of scope")
    return m


def supersede_memory(
    memory_id: str, *, superseded_by: str | None = None, reason: str | None = None
) -> dict:
    """Bi-temporally supersede (soft-retire) one memory — NO hard delete.

    Sets ``valid_to = now(UTC)``, ``is_active = False``, and, when ``superseded_by``
    is given, ``superseded_by_id`` to that (also in-scope) memory. The row stays in
    the table (auditable); recall no longer returns it (``is_active`` filter). See
    the module docstring for how this manual retire differs from production's
    automatic conflict-supersession (which keeps ``is_active=True``).

    Returns ``{"id", "superseded_by", "valid_to"}``. Raises ``ValueError`` if
    ``memory_id`` (or ``superseded_by``) does not exist / is out of scope.
    """
    db = get_session()
    try:
        m = _get_in_scope(db, memory_id)
        superseder = _get_in_scope(db, superseded_by) if superseded_by is not None else None
        now = datetime.now(UTC)
        m.valid_to = now
        m.is_active = False
        if superseder is not None:
            m.superseded_by_id = superseder.id
        if reason:
            logger.info("supersede_memory %s (reason=%r, not persisted)", memory_id, reason)
        db.commit()
        return {
            "id": str(m.id),
            "superseded_by": str(superseder.id) if superseder is not None else None,
            "valid_to": now.isoformat(),
        }
    finally:
        db.close()


def _dedup_ids(memory_ids: list[str]) -> list[str]:
    """Distinct ids, order-preserving (a UI double-select is not two children)."""
    seen: set = set()
    out: list[str] = []
    for mid in memory_ids:
        key = str(mid)
        if key not in seen:
            seen.add(key)
            out.append(mid)
    return out


def merge_memories(memory_ids: list[str], *, summary: str | None = None) -> dict:
    """Consolidate >=2 memories into one new ACTIVE parent — a MANUAL user merge.

    Reuses the SAME ``consolidation.insert_root_archive_sources`` primitive
    production's AUTOMATIC union-merge uses (no fork), with ``supersede_children=True``
    so a user-driven merge records the fuller audit trail. The children must SHARE
    scope (``user_id``, ``agent_type``, ``project_id``, ``customer_id``) — a
    mixed-scope merge raises ``ValueError``. The parent's ``context_summary`` is the
    provided ``summary`` else the shared deterministic union of the children's
    summaries. The parent inherits identity from the newest child, backdates
    ``created_at``/``occurred_at`` to the earliest child, is re-embedded via the
    shared ``store`` embedding path, and carries the union of the children's tags.
    Each child is then archived + superseded: ``is_active=False``,
    ``parent_id=parent``, ``valid_to=now``, ``superseded_by_id=parent``.

    Returns ``{"parent_id", "merged_count", "child_ids"}``.
    """
    ids = _dedup_ids(list(memory_ids or []))
    if len(ids) < 2:
        raise ValueError("merge_memories requires at least 2 distinct memory ids")

    db = get_session()
    try:
        children = [_get_in_scope(db, mid) for mid in ids]
        scopes = {
            (c.user_id, c.agent_type, c.project_id, c.customer_id) for c in children
        }
        if len(scopes) != 1:
            raise ValueError(
                "cannot merge memories from mixed scopes — user_id, agent_type, "
                "project_id and customer_id must all match across the children"
            )

        # Deterministic order (production reads its pool ordered by (created_at, id)).
        children.sort(key=lambda c: (c.created_at, str(c.id)))
        text = (
            summary.strip()
            if (summary and summary.strip())
            else deterministic_union_summary(children)
        )
        child_ids = [str(c.id) for c in children]
        root = insert_root_archive_sources(db, text, children, supersede_children=True)
        db.commit()
        return {
            "parent_id": str(root.id),
            "merged_count": len(children),
            "child_ids": child_ids,
        }
    finally:
        db.close()


def reset_db() -> dict:
    """Empty the three memory tables — the UI "Reset DB" button. Idempotent.

    Deletes ``memory_tag_link`` first (its composite PK FKs both other tables),
    then ``agent_memory`` (self-FKs are ``SET NULL``), then ``memory_tag``. Uses
    DELETE, not TRUNCATE, so it runs on the SQLite-degraded path, returns row
    counts, and is safely re-runnable (a second call deletes nothing). The harder
    CLI ``run_demo.sh reset`` wipes the whole cluster; this is the softer in-DB one.

    Returns ``{"deleted_memories", "deleted_tags"}``.
    """
    db = get_session()
    try:
        n_mem = db.query(AgentMemory).count()
        n_tag = db.query(MemoryTag).count()
        db.query(MemoryTagLink).delete(synchronize_session=False)
        db.query(AgentMemory).delete(synchronize_session=False)
        db.query(MemoryTag).delete(synchronize_session=False)
        db.commit()
        return {"deleted_memories": n_mem, "deleted_tags": n_tag}
    finally:
        db.close()
