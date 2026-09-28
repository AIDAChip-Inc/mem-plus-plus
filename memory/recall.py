"""In-process recall + write API — ported from production ``scripts/team_memory.py``.

The public surface Qiyas/eval code consumes:

    recall_facts(persona, query, k=24) -> list[dict{summary, occurred_at, hit_count, by}]
    store_facts(persona, content, mode="auto") -> dict
    store_facts_verbatim(persona, facts) -> dict

Scope (contract): ``project_id = uuid5(NS, "project:memory-research")``,
``user_id = uuid5(NS, "user:" + (MEMORY_USER or OS login))``, ``agent_type = persona``.
A deterministic ``customer_id = uuid5(NS, "customer:memory-research")`` is also
derived so ``recall_project_sections``' TEAM (cross-persona) pass runs — matching
production's two-section recall.

Trimmed vs production: no Railway/remote DSN, no 1Password/self-heal, no
retry/pool-reset, no daemon/AF_UNIX socket, no LLM auth gate, no schema self-heal,
and no ``_ensure_scope_user`` (scope columns are plain GUIDs here, no user FK to
provision). ``store_facts`` honors ``config.MEMORY_ATOMIC_FACTS`` exactly as
production does: OFF (the production default) => the gated single-summary path.
"""
from __future__ import annotations

import getpass
import logging
import os
import uuid
from datetime import UTC, date, datetime

from memory import config
from memory.db import get_session
from memory.extraction import (
    MemoryExtractionUnavailable,
    extract_atomic_facts,
    extract_summary_and_tags,
    extract_two_section,
    normalize_label,
)
from memory.models import AgentMemory
from memory.store import PostgresMemoryStore

logger = logging.getLogger(__name__)

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "memory-research.aidachip.com")
_BY_PREFIX = "by:"


def _contributor_slug() -> str:
    """Per-human authorship slug: ``MEMORY_USER`` env, else the OS login.

    Reuses ``extraction.normalize_label`` (same lowercase/collapse rule the tags
    are stored under) so a human's recall scope and their ``by:<slug>`` tag agree.
    """
    try:
        who = (os.environ.get("MEMORY_USER") or getpass.getuser() or "unknown").strip()
    except Exception:
        who = "unknown"
    return normalize_label(who) or "unknown"


def _scope() -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, str]:
    """Return ``(user_id, project_id, customer_id, contributor_slug)``."""
    slug = _contributor_slug()
    uid = uuid.uuid5(_NS, f"user:{slug}")
    pid = uuid.uuid5(_NS, "project:memory-research")
    cid = uuid.uuid5(_NS, "customer:memory-research")
    return uid, pid, cid, slug


def _contributors(m: AgentMemory) -> str:
    """Join a memory's ``by:<who>`` authorship tags (sorted, comma-joined). MUST run
    inside the owning session — ``m.tags`` is lazy="selectin"."""
    names = sorted(
        who
        for t in (m.tags or [])
        if t.label.startswith(_BY_PREFIX) and (who := t.label[len(_BY_PREFIX):])
    )
    return ", ".join(names)


def _materialize(mems: list[AgentMemory]) -> list[dict]:
    """ORM rows -> plain dicts. MUST run inside the owning session (commit expires
    instances; ``m.tags`` is lazy-loaded)."""
    rows: list[dict] = []
    for m in mems:
        row: dict = {
            "summary": (m.context_summary or m.content or "").strip(),
            "occurred_at": m.occurred_at.date().isoformat() if m.occurred_at else None,
            "hit_count": m.hit_count,
        }
        by = _contributors(m)
        if by:
            row["by"] = by
        rows.append(row)
    return rows


def recall_facts(persona: str, query: str, k: int = 24, update_hits: bool = True) -> list[dict]:
    """Scope-filtered, salience-ranked recall -> plain dicts (session-safe).

    OWN is this human's personal recall (per-human user_id + this persona's
    agent_type). TEAM is the project pool — every human/persona on the project
    EXCEPT the caller's own scope. OWN renders first, TEAM appended; each row
    carries its own ``by:<slug>`` tag so a teammate's fact is distinguishable.
    """
    uid, pid, cid, _slug = _scope()
    db = get_session()
    try:
        store = PostgresMemoryStore(db)
        own, own_plan, team, team_plan = store.recall_project_sections(
            user_id=uid, agent_type=persona, project_id=pid, customer_id=cid, query=query, k=k
        )
        mems = own + team
        if update_hits and (own_plan or team_plan):
            store.record_recall_hits(
                own_plan + team_plan, agent_type=persona, refresh_instances=True
            )
        return _materialize(mems)
    finally:
        db.close()


def _persist_facts(
    persona: str,
    facts: list[tuple[str, list[str], datetime | None]],
    mode: str,
    degraded: dict,
    content_blob: str | None = None,
    agent_summary: str | None = None,
) -> dict:
    """Insert pre-built ``(summary, tags, occurred_at)`` facts as N rows via the
    shared ``store.write`` path. The row ``content`` is ``content_blob`` when given
    (the whole blob) else the fact's own summary (storage-only path — embedding
    matches the summary). Light exact-summary dedup in-scope (idempotent on retry).

    ``agent_summary`` (two-section cell, M5) is the tight summary of what the AGENT
    said this turn — recall PAYLOAD only, never embedded/tsv'd (``store.write``
    excludes it). It is a per-TURN value, so it is threaded only by the gated
    two-section path, which emits exactly one fact; the atomic/verbatim paths pass
    ``None`` (production populates it on the two-section write path alone).
    """
    if not facts:
        return {"written": 0, "mode": mode, "reason": "nothing-worth-storing", **degraded}

    uid, pid, cid, slug = _scope()
    contrib = f"{_BY_PREFIX}{slug}"
    db = get_session()
    n = chars = 0
    try:
        store = PostgresMemoryStore(db)
        for summary, tags, occ in facts:
            if not summary.strip():
                continue
            dup = (
                db.query(AgentMemory.id)
                .filter(
                    AgentMemory.user_id == uid,
                    AgentMemory.agent_type == persona,
                    AgentMemory.project_id == pid,
                    AgentMemory.context_summary == summary,
                    AgentMemory.is_active.is_(True),
                )
                .first()
            )
            if dup is not None:
                continue
            store.write(
                user_id=uid,
                agent_type=persona,
                project_id=pid,
                customer_id=cid,
                content=content_blob if content_blob is not None else summary,
                context_summary=summary,
                agent_summary=agent_summary,
                tags=list(tags or []) + [contrib],
                occurred_at=occ,
                session_id=None,
                source_message_id=None,
            )
            n += 1
            chars += len(summary)
    finally:
        db.close()
    return {"written": n, "written_chars": chars, "mode": mode, **degraded}


def store_facts(persona: str, content: str, mode: str = "auto", model: str | None = None) -> dict:
    """Extract (LLM) + store facts for a persona. Mirrors production extract_and_store.

    ``mode`` ``"auto"`` resolves to ``"atomic"`` iff ``config.MEMORY_ATOMIC_FACTS``
    (production default OFF => ``"gated"`` single-summary). ``model`` (friendly name
    haiku/sonnet/opus OR a full id) is threaded to the post-hook extractor; ``None``
    keeps the Haiku default, so behavior is unchanged when it is not passed
    (``store_facts_verbatim`` is LLM-free and takes no model). On an extraction OUTAGE
    (no LLM / unparseable), the result carries ``degraded: True`` + a ``reason``:
    the atomic path falls back to the deterministic summary path; the gated path
    degrades visibly but stores nothing.
    """
    if not content or not content.strip():
        return {"written": 0, "mode": mode, "reason": "empty"}

    if mode == "auto":
        mode = "atomic" if config.MEMORY_ATOMIC_FACTS else "gated"

    degraded_reason = None
    agent_summary = None
    if mode == "atomic":
        try:
            facts = extract_atomic_facts(content, raise_on_unavailable=True, model=model)
        except MemoryExtractionUnavailable as exc:
            degraded_reason = f"atomic-extraction-unavailable ({exc}); summary-path fallback"
            summary, tags, occ = extract_summary_and_tags(content, gate=False, model=model)
            facts = [(summary, tags, occ)] if summary.strip() else []
    else:  # "gated" (default) or "summary" (ungated)
        # Two-section capture (M5): the LIVE gated path also extracts agent_summary
        # when MEMORY_TWO_SECTION is ON (production default). Mirrors production's
        # extract_turn_facts dispatch. Ungated "summary" mode keeps the plain path.
        two_section = mode == "gated" and config.MEMORY_TWO_SECTION
        try:
            if two_section:
                summary, tags, occ, agent_summary = extract_two_section(
                    content, raise_on_unavailable=True, model=model
                )
            else:
                summary, tags, occ = extract_summary_and_tags(
                    content, gate=(mode == "gated"), raise_on_unavailable=True, model=model
                )
        except MemoryExtractionUnavailable as exc:
            degraded_reason = f"gated-extraction-unavailable ({exc}); turn not stored"
            summary, tags, occ, agent_summary = "", [], None, None
        facts = [(summary, tags, occ)] if summary.strip() else []

    degraded = {"degraded": True, "reason": degraded_reason} if degraded_reason else {}
    return _persist_facts(
        persona, facts, mode, degraded, content_blob=content, agent_summary=agent_summary
    )


def _coerce_occurred_at(value: object) -> datetime | None:
    """Normalize an ``occurred_at`` to ``datetime | None``, faithful to production
    ``team_memory.py``: ``None``/``datetime`` pass through; an ISO **date** string
    becomes midnight-UTC; an ISO **datetime** string is parsed as-is.
    """
    if value is None or isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.combine(date.fromisoformat(raw), datetime.min.time(), tzinfo=UTC)
    except ValueError:
        return datetime.fromisoformat(raw)


def _normalize_verbatim_fact(fact: object) -> tuple[str, list[str], datetime | None]:
    """Accept any of the three input shapes INTERFACE_CONTRACT §3 allows (plus a
    bare string, a research-tool convenience) and return the canonical
    ``(summary, tags, occurred_at)`` 3-tuple ``_persist_facts`` consumes:

    * ``str`` → summary only, no tags, no date.
    * ``dict`` → ``{"summary", "tags"?, "occurred_at"?}`` (occurred_at ISO or None).
    * ``(summary, tags, occurred_at)`` → occurred_at may be a datetime OR ISO string.
    """
    if isinstance(fact, str):
        return (fact, [], None)
    if isinstance(fact, dict):
        return (
            str(fact.get("summary") or ""),
            list(fact.get("tags") or []),
            _coerce_occurred_at(fact.get("occurred_at")),
        )
    summary, tags, occ = fact  # (summary, tags, occurred_at)
    return (str(summary or ""), list(tags or []), _coerce_occurred_at(occ))


def store_facts_verbatim(persona: str, facts: list) -> dict:
    """Storage-only write (session-close): persist ALREADY-atomized facts as N rows
    with NO extraction, NO LLM, NO auth dependency — so it CANNOT degrade on an LLM
    outage. Each fact's OWN text becomes the row ``content`` (embedding matches the
    summary), so recall behaves identically to the atomic-extraction path.

    Per INTERFACE_CONTRACT §3, each fact may be the canonical dict form, a
    ``(summary, tags, occurred_at)`` tuple, or a bare summary string; ISO
    ``occurred_at`` values are coerced to ``datetime`` (see ``_coerce_occurred_at``).
    """
    normalized = [_normalize_verbatim_fact(f) for f in facts]
    return _persist_facts(persona, normalized, "facts", {})
