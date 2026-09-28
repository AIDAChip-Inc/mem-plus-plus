"""Read-only browse/query API over ``agent_memory`` — for the "Memory" UI tab.

This is a SIBLING to ``recall.py``, not a replacement: recall ranks a scope slice
by salience (MATCH -> HITS -> RECENCY) *and writes a hit back* so recall stats
reflect real usage. Browsing is different — a human paging through the DB in a UI
must NOT perturb those stats. So ``browse_memories`` is **strictly read-only**: it
opens a session, runs ONE ``SELECT`` with the caller's filters, materializes plain
dicts, and closes — it never calls ``record_recall_hits``, never issues an
``UPDATE``, and never commits. ``hit_count``/``last_used_at``/``salience`` are left
exactly as recall left them.

Scope (own/team/all) reuses ``recall._scope`` (the uuid5 project/user derivation),
so browse sees exactly the project pool recall operates over. Browse exposes two
ORTHOGONAL axes (unlike recall, which fuses them into one two-section split):

* ``persona`` — an ``agent_type`` filter (the ROLE axis): which persona authored the
  memory. Applies in every scope; ``None`` means all personas.
* ``scope`` — the USER axis (``own``/``team``/``all``): ``own`` = this human's rows
  (``user_id == uid``), ``team`` = teammates' rows (``user_id != uid``), ``all`` =
  the whole project pool. In this single-human replica ``team`` is usually empty —
  that is a faithful reflection of the data, not a bug; the axis is correct for the
  general multi-human case.

The per-row ``scope`` label is ``own``/``team`` on the same user_id test.

The returned ``tags`` list has the ``by:<slug>`` authorship tags SEPARATED OUT into
a ``by`` field (comma-joined, via ``recall._contributors``), so ``tags`` holds only
the semantic entity tags — matching how the recall payload treats authorship.

Both ``browse_memories`` (one dict per row) and ``get_memory`` (single row) return
the **full production cell** — EVERY ``AgentMemory`` column, JSON-safe — via one
shared ``_full_cell`` materializer, so the two never drift and the UI can render the
whole cell as inline table columns. The 384-d embedding is reported by presence +
dim only (``has_embedding`` / ``embedding_dim``), never dumped as raw floats; the
full-text vector likewise by presence (``has_tsv``).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sqlalchemy import func, or_, select

from memory import config
from memory.db import get_session
from memory.embeddings import get_embedding_service
from memory.extraction import normalize_label
from memory.models import AgentMemory, MemoryTag, MemoryTagLink
from memory.recall import _BY_PREFIX, _coerce_occurred_at, _contributors, _scope
from memory.store import _TS_CONFIG, _query_terms, _with_time_bounds

logger = logging.getLogger(__name__)

_ORDERS = frozenset({"recency", "hits", "occurred"})
_SCOPES = frozenset({"own", "team", "all"})


def _is_postgres(db) -> bool:
    return db.get_bind().dialect.name == "postgresql"


def _scope_filters(scope: str, persona: str | None, uid, pid, cid) -> list:
    """Project-pool base + (independent) persona filter + own/team USER narrowing."""
    filters = [
        AgentMemory.project_id == pid,
        AgentMemory.customer_id == cid,
        AgentMemory.is_active.is_(True),
    ]
    if persona is not None:
        filters.append(AgentMemory.agent_type == persona)
    if scope == "own":
        filters.append(AgentMemory.user_id == uid)
    elif scope == "team":
        filters.append(AgentMemory.user_id != uid)
    return filters


def _row_scope(m: AgentMemory, uid) -> str:
    """Per-row own/team label on the USER axis (this human vs. teammates)."""
    return "own" if m.user_id == uid else "team"


def _apply_lexical(q, db, text: str):
    """Lexical text FILTER — PG full-text (``@@`` websearch), else ILIKE, mirroring
    how ``store._recall_scoped`` degrades. A filter (not a ranking leg): any-term
    match on SQLite; a raw-substring fallback when the query has no usable terms."""
    if _is_postgres(db):
        ts_query = func.websearch_to_tsquery(_TS_CONFIG, text)
        return q.filter(AgentMemory.content_tsv.isnot(None)).filter(
            AgentMemory.content_tsv.op("@@")(ts_query)
        )
    patterns = [f"%{t}%" for t in _query_terms(text)] or [f"%{text.strip()}%"]
    clauses = []
    for pattern in patterns:
        clauses.append(AgentMemory.content.ilike(pattern))
        clauses.append(AgentMemory.context_summary.ilike(pattern))
    return q.filter(or_(*clauses))


def _semantic_qvec(db, text: str):
    """Embed ``text`` with the SAME MiniLM embedder recall/store use. Returns None
    (caller degrades to lexical) when not on PG, embeddings are off, or the embedder
    cannot load — never raises."""
    if not _is_postgres(db) or not config.MEMORY_EMBEDDINGS_ENABLED:
        return None
    try:
        return get_embedding_service("memory").embed(text, intent="query")
    except Exception as exc:  # embedder/model load failure — degrade, don't crash
        logger.warning("browse semantic embed failed (%s) — degrading to lexical", exc)
        return None


def _apply_order(q, order: str):
    if order == "hits":
        return q.order_by(AgentMemory.hit_count.desc(), AgentMemory.created_at.desc())
    if order == "occurred":
        return q.order_by(
            AgentMemory.occurred_at.desc().nullslast(), AgentMemory.created_at.desc()
        )
    return q.order_by(AgentMemory.created_at.desc())  # "recency" (default)


def _full_cell(m: AgentMemory, uid) -> dict:
    """ORM row -> the FAITHFUL full production cell: EVERY ``AgentMemory`` column,
    JSON-safe (GUIDs/datetimes stringified). This is the SINGLE materializer shared
    by ``browse_memories`` (one dict per row, so the UI can render the whole cell as
    inline table columns) and ``get_memory`` (single-row fetch) — so the two can
    never drift. Authorship ``by:`` tags are split out of ``tags`` into ``by`` (as
    browse always did); the 384-d embedding is reported by PRESENCE + DIM only, never
    dumped as raw floats. MUST run inside the owning session — ``m.tags`` is
    lazy="selectin".
    """
    tags = sorted(t.label for t in (m.tags or []) if not t.label.startswith(_BY_PREFIX))
    has_embedding = m.embedding is not None
    return {
        # identity / content
        "id": str(m.id),
        "summary": (m.context_summary or m.content or "").strip(),
        "content": m.content,
        "context_summary": m.context_summary,
        "agent_summary": m.agent_summary,
        # tags (semantic) + authorship split out into `by`
        "tags": tags,
        "by": _contributors(m),
        # scope coordinates
        "user_id": str(m.user_id) if m.user_id else None,
        "agent_type": m.agent_type,
        "project_id": str(m.project_id) if m.project_id else None,
        "customer_id": str(m.customer_id) if m.customer_id else None,
        "session_id": str(m.session_id) if m.session_id else None,
        "scope": _row_scope(m, uid),
        # classification
        "memory_tier": m.memory_tier,
        "discipline": m.discipline,
        "authority": m.authority,
        # temporal
        "occurred_at": m.occurred_at.isoformat() if m.occurred_at else None,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None,
        "valid_from": m.valid_from.isoformat() if m.valid_from else None,
        "valid_to": m.valid_to.isoformat() if m.valid_to else None,
        # lifecycle
        "is_active": m.is_active,
        "superseded_by_id": str(m.superseded_by_id) if m.superseded_by_id else None,
        "parent_id": str(m.parent_id) if m.parent_id else None,
        # ranking
        "hit_count": m.hit_count,
        "last_hit_by": m.last_hit_by,
        "salience": m.salience,
        # provenance
        "source_message_id": str(m.source_message_id) if m.source_message_id else None,
        # search artifacts — PRESENCE + DIM only, never the raw 384 floats
        "has_embedding": has_embedding,
        "embedding_dim": len(m.embedding) if has_embedding else None,
        "has_tsv": m.content_tsv is not None,
    }


def browse_memories(
    *,
    persona: str | None = None,
    text: str | None = None,
    tags: list[str] | None = None,
    occurred_after: datetime | str | None = None,
    occurred_before: datetime | str | None = None,
    min_hit_count: int = 0,
    scope: str = "all",
    order: str = "recency",
    semantic: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Read-only, filterable browse over ``agent_memory`` — the UI's query API.

    Filters (all optional, AND-combined): ``persona`` -> ``agent_type`` (role axis);
    ``tags`` -> ANY-match via the tag link table (labels normalized);
    ``occurred_after/before`` -> ``occurred_at`` range (accepts datetimes or ISO
    strings); ``min_hit_count`` -> ``hit_count >=``; ``scope`` -> own/team/all on the
    USER axis (own = this human, team = teammates, all = whole project pool);
    ``is_active=True`` is always on. ``persona`` and ``scope`` are orthogonal.

    ``text`` with ``semantic=False`` (default) is a LEXICAL filter (PG full-text,
    else ILIKE). ``text`` with ``semantic=True`` ranks the scope by pgvector cosine
    similarity to the embedded text (``order`` is then ignored); it degrades to the
    lexical filter + ``order`` when embeddings/pgvector are unavailable.

    ``order``: ``recency`` (created_at desc) | ``hits`` (hit_count desc) |
    ``occurred`` (occurred_at desc, nulls last). ``limit``/``offset`` paginate.

    READ-ONLY: never bumps ``hit_count`` and never writes — browsing must not
    perturb recall stats. Returns plain dicts (session-safe).
    """
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {sorted(_SCOPES)}, got {scope!r}")
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {sorted(_ORDERS)}, got {order!r}")

    uid, pid, cid, _slug = _scope()
    filters = _scope_filters(scope, persona, uid, pid, cid)
    _with_time_bounds(
        filters, _coerce_occurred_at(occurred_after), _coerce_occurred_at(occurred_before)
    )
    if min_hit_count > 0:
        filters.append(AgentMemory.hit_count >= min_hit_count)

    db = get_session()
    try:
        q = db.query(AgentMemory).filter(*filters)

        if tags:
            labels = [lbl for t in tags if (lbl := normalize_label(t))]
            if labels:
                tagged = (
                    select(MemoryTagLink.memory_id)
                    .join(MemoryTag, MemoryTag.id == MemoryTagLink.tag_id)
                    .where(MemoryTag.label.in_(labels))
                )
                q = q.filter(AgentMemory.id.in_(tagged))

        has_text = bool(text and text.strip())
        semantic_active = False
        if semantic and has_text:
            qvec = _semantic_qvec(db, text)
            if qvec is not None:
                q = q.filter(AgentMemory.embedding.isnot(None)).order_by(
                    AgentMemory.embedding.cosine_distance(qvec)
                )
                semantic_active = True
        if not semantic_active:
            if has_text:
                q = _apply_lexical(q, db, text)
            q = _apply_order(q, order)

        rows = q.offset(max(offset, 0)).limit(max(limit, 0)).all()
        return [_full_cell(m, uid) for m in rows]
    finally:
        db.close()


def get_memory(memory_id: str) -> dict | None:
    """Read-only full-cell read: the EXACT ``agent_memory`` row as production defines
    it, or ``None`` if not found / out of scope.

    Returns EVERY column of the ``AgentMemory`` row (see ``_full_cell``) — the
    faithful full production cell — so a UI can show the whole cell for one memory.
    Scope is resolved exactly as ``browse_memories`` does (``recall._scope`` — the
    same project/customer pool): a row in a different project pool returns ``None``.
    Unlike ``browse_memories`` this does NOT require ``is_active`` — a superseded or
    archived cell can still be inspected by its id.

    STRICTLY READ-ONLY: one ``SELECT``, no ``record_recall_hits``, no ``UPDATE``, no
    commit — ``hit_count`` / ``last_used_at`` / ``salience`` are never perturbed. The
    384-d embedding is reported by presence + dim only, never dumped as raw floats.
    """
    try:
        mid = uuid.UUID(str(memory_id))
    except (ValueError, TypeError):
        return None  # not a valid GUID -> cannot be a real cell

    uid, pid, cid, _slug = _scope()
    db = get_session()
    try:
        m = (
            db.query(AgentMemory)
            .filter(
                AgentMemory.id == mid,
                AgentMemory.project_id == pid,
                AgentMemory.customer_id == cid,
            )
            .one_or_none()
        )
        return _full_cell(m, uid) if m is not None else None
    finally:
        db.close()
