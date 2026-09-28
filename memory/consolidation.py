"""Synthesize-merge consolidation — faithful replica of production consolidation.

Two layers, mirroring production but co-located here (the replica keeps ``store.py``
a pure recall+write core, so the DB-facing consolidation orchestration production
puts in ``store.consolidate_*`` lives here instead):

1. **DB-free helpers** — ported verbatim-in-spirit from production
   ``app/services/memory/consolidation.py`` (validated τ=0.85: LoCoMo +0.011, LME
   neutral, OrgMemBench recall +0.027): ``dedup_groups`` (greedy disjoint near-dup
   grouping, cosine ≥ τ), ``MERGE_PROMPT``, ``build_facts_block``, ``DEFAULT_TAU``,
   ``ConsolidationResult``, ``NEVER_MERGE_LABELS``, and the supersession detection
   seam (``GroupRelation`` + ``classify_relation`` + the ``CONFLICT_DETECT_*``
   prompt/schema). Zero DB / zero framework coupling.

2. **DB-facing orchestration** — ``consolidate_scope`` / ``_consolidate_pool``
   (faithful to production ``store.consolidate_scope`` / ``_consolidate_pool``),
   the shared ``insert_root_archive_sources`` / ``supersede_conflict`` /
   ``never_merge_ids`` primitives, and the MANUAL ``consolidate(persona=…)``
   entry-point.

**Manual entry-point, NOT the production background loop (Nizam, 2026-07-23).**
Production runs consolidation from an asyncio loop guarded by a Postgres advisory
lock + leader guard + a durable due-gate (``main.py``). Those are multi-instance
OPS infra — the same class as the daemon/Railway/Flagsmith the replica deliberately
strips. ``consolidate(persona)`` here is a deterministic, on-demand call (invocable
from the eval harness and a "Consolidate now" UI button); it records a
``MemoryConsolidationRun`` ledger row but takes no lock and runs no scheduler.

**Degradable synthesis (M1).** With an LLM available (``llm_available``) the pass is
byte-faithful to production: Haiku union-synthesis per group, and — when
``MEMORY_SUPERSESSION`` is ON — LLM classification routing CONFLICT groups to
deterministic newest-wins supersession. With NO LLM the pass degrades to a
deterministic lossless union summary (``deterministic_union_summary``, the same
primitive ``mutate.merge_memories`` uses) and skips supersession classification.
This is a DOCUMENTED manual/offline difference: production skips a group on an LLM
failure (never archiving what it couldn't merge), whereas the offline replica
still merges near-dups deterministically so ``consolidate()`` is useful without a key.

Reuse (no fork): the root insert re-embeds via ``store._populate_search_fields``
(the exact path ``write`` uses); ``insert_root_archive_sources`` /
``deterministic_union_summary`` are shared with ``mutate.merge_memories`` so the
manual and automatic merges never fork two implementations.
"""
from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from memory import config
from memory.db import get_session
from memory.embeddings import first_nonblank
from memory.llm import llm_available, llm_complete, structured_call
from memory.models import AgentMemory, MemoryConsolidationRun, MemoryTag, MemoryTagLink
from memory.store import PostgresMemoryStore, make_scope_key

logger = logging.getLogger(__name__)

# The default near-duplicate threshold — the validated shrink lever.
DEFAULT_TAU = 0.85

# Per-GROUP prompt budget (chars), NOT a per-fact cap: facts are sent whole so the
# "preserve every distinct detail" contract holds; the cap only bounds total prompt
# size for a pathologically large group.
GROUP_PROMPT_BUDGET = 12000

# Haiku union-synthesis prompt (verbatim-token guardrail): the union-of-detail
# property depends on preserving numbers/IDs/hosts/dates/error codes verbatim and
# abstracting only prose.
MERGE_PROMPT = (
    "You are consolidating an agent's memory. The facts below are near-duplicates of the "
    "SAME underlying fact. Rewrite them as ONE concise fact that preserves EVERY distinct "
    "detail VERBATIM — names, numbers, dates, ids, hostnames, error codes, version pins, "
    "and qualifiers — abstracting only connective prose. Invent nothing and drop nothing. "
    "Output ONLY the merged fact text, no preamble.\n\n{facts}\n\nMerged fact:"
)


class GroupRelation(enum.StrEnum):
    """How a near-dup group relates (supersession detection).

    Only ``CONFLICT`` (same attribute, diverging value) takes the deterministic
    newest-wins supersession path; ``RESTATEMENT`` (same fact re-worded) keeps the
    union-merge; ``DISTINCT_FACTS`` (grouped by embedding proximity but not the
    same fact) is left untouched.
    """

    CONFLICT = "conflict"
    RESTATEMENT = "restatement"
    DISTINCT_FACTS = "distinct_facts"


# Detection = LLM structured/tool output (team standard: never regex-on-prose).
CONFLICT_DETECT_TOOL = "classify_memory_group"
CONFLICT_DETECT_TOOL_DESC = (
    "Classify how a group of near-duplicate agent memories relates."
)
CONFLICT_DETECT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": [r.value for r in GroupRelation],
            "description": (
                "conflict = the facts assert DIFFERENT values for the SAME "
                "attribute (a value changed over time); restatement = the SAME "
                "fact re-worded with no value disagreement; distinct_facts = "
                "related but genuinely different facts."
            ),
        },
    },
    "required": ["relation"],
    "additionalProperties": False,
}
CONFLICT_DETECT_PROMPT = (
    "Below is a group of near-duplicate agent memories. Decide their relation:\n"
    "- conflict: they state DIFFERENT values for the SAME attribute (e.g. a "
    "spec value, count, date, or status that changed over time).\n"
    "- restatement: they express the SAME fact with the same values, only "
    "re-worded.\n"
    "- distinct_facts: they are related but genuinely different facts.\n\n"
    "{facts}\n\nClassify the group."
)

# Tag labels that mark a memory as never-merge: confidential (project-locked) or
# pinned (must-keep-verbatim). Rows carrying either are excluded from the candidate
# pool entirely, so they are never archived or merged.
NEVER_MERGE_LABELS: frozenset[str] = frozenset({"confidential", "pinned"})


def classify_relation(result: dict | None) -> GroupRelation:
    """Map a ``structured_call`` result to a ``GroupRelation``.

    A missing/None result (detection failure, no client) or an unrecognized value
    defaults to ``DISTINCT_FACTS`` — the safe fallback: leave the group UNTOUCHED
    on uncertain detection (forgoes a dedup opportunity; never mutates).
    """
    if not isinstance(result, dict):
        return GroupRelation.DISTINCT_FACTS
    try:
        return GroupRelation(result.get("relation"))
    except ValueError:
        return GroupRelation.DISTINCT_FACTS


@dataclass
class ConsolidationResult:
    """Outcome of one ``consolidate_scope`` pass."""

    groups: int = 0          # candidate groups of size >= 2 found
    synthesized: int = 0     # roots successfully inserted (merges applied)
    archived: int = 0        # source leaves archived under a root
    cost_calls: int = 0      # LLM calls issued (cost)
    errors: int = 0          # groups left untouched by a merge failure
    superseded: int = 0      # older rows marked valid_to (kept recallable)

    def add(self, other: ConsolidationResult) -> None:
        """Accumulate another scope's result into this aggregate (in place)."""
        self.groups += other.groups
        self.synthesized += other.synthesized
        self.archived += other.archived
        self.cost_calls += other.cost_calls
        self.errors += other.errors
        self.superseded += other.superseded


def _norm(emb) -> np.ndarray:
    """L2-normalize an embedding (accepts a list or a pgvector string form)."""
    if isinstance(emb, str):
        emb = [float(x) for x in emb.strip("[]").split(",")]
    v = np.asarray(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def dedup_groups(embeddings, tau: float = DEFAULT_TAU) -> list[list[int]]:
    """Greedy disjoint near-dup groups (cosine >= tau). Each item in <=1 group.

    Returns index groups of size >= 2 (singletons are not merge candidates).
    Greedy + disjoint (``seen[]``) ⇒ a freshly-synthesized fact is never re-merged
    in the same pass — the within-pass convergence guarantee.
    """
    vecs = [_norm(e) for e in embeddings]
    seen = [False] * len(vecs)
    groups: list[list[int]] = []
    for i in range(len(vecs)):
        if seen[i]:
            continue
        grp = [i]
        seen[i] = True
        for j in range(i + 1, len(vecs)):
            if not seen[j] and float(vecs[i] @ vecs[j]) >= tau:
                grp.append(j)
                seen[j] = True
        if len(grp) >= 2:
            groups.append(grp)
    return groups


def build_facts_block(texts: list[str]) -> str:
    """Bullet the FULL fact texts — each fact is sent WHOLE, never clipped.

    Union-of-detail requires every ID/date/host/error-code survive to synthesis,
    so a fact is never truncated. ``GROUP_PROMPT_BUDGET`` is a soft target: near-dup
    groups are small in practice, so exceeding it is pathological — we log it and
    still emit the complete facts rather than silently drop tail detail.
    """
    bullets = [f"- {t}" for t in texts]
    block = "\n".join(bullets)
    if len(block) > GROUP_PROMPT_BUDGET:
        logger.warning(
            "Consolidation group exceeds prompt budget; sending whole (lossless)",
            extra={"chars": len(block), "budget": GROUP_PROMPT_BUDGET, "facts": len(bullets)},
        )
    return block


def deterministic_union_summary(sources) -> str:
    """Offline, LLM-free stand-in for Haiku union-synthesis (shared with
    ``mutate.merge_memories``).

    Joins the sources' DISTINCT summaries (``context_summary`` else ``content``) in
    order with ``" ; "`` — every distinct detail survives losslessly, which is the
    property ``MERGE_PROMPT`` targets. ``sources`` is expected pre-sorted by
    ``(created_at, id)`` for determinism.
    """
    texts: list[str] = []
    for s in sources:
        t = (first_nonblank(s.context_summary, s.content) or "").strip()
        if t and t not in texts:
            texts.append(t)
    return " ; ".join(texts)


# ── DB-facing primitives (faithful to production store.py methods) ──────────────

def never_merge_ids(db, pool_filters) -> set:
    """Ids in the pool carrying a never-merge tag (``confidential``/``pinned``).

    Joined through the memory rows themselves, so a tagged row from ANY agent/user
    in a project pool is excluded — the per-scope tag registry's ``scope_key`` never
    bounds the exclusion. Excluded rows never enter the candidate pool.
    """
    rows = (
        db.query(MemoryTagLink.memory_id)
        .join(MemoryTag, MemoryTag.id == MemoryTagLink.tag_id)
        .join(AgentMemory, AgentMemory.id == MemoryTagLink.memory_id)
        .filter(MemoryTag.label.in_(NEVER_MERGE_LABELS), *pool_filters)
        .all()
    )
    return {r[0] for r in rows}


def classify_group(texts: list[str]) -> GroupRelation:
    """Detection seam: LLM structured/tool classification of a near-dup group.

    Uses the forced-tool ``structured_call``; a None result (no client / failure)
    or an unknown value degrades to ``DISTINCT_FACTS`` via :func:`classify_relation`
    — the safe default that never supersedes on uncertain detection.
    """
    result = structured_call(
        CONFLICT_DETECT_PROMPT.format(facts=build_facts_block(texts)),
        tool_name=CONFLICT_DETECT_TOOL,
        tool_description=CONFLICT_DETECT_TOOL_DESC,
        input_schema=CONFLICT_DETECT_SCHEMA,
        model="haiku",
    )
    return classify_relation(result)


def supersede_conflict(db, sources) -> int:
    """Deterministic newest-wins supersession for one value-conflict group.

    Faithful to production ``store._supersede_conflict``: the current fact is the
    one with the latest EVENT time (``valid_from`` = occurred_at else created_at);
    it stays as-is (``valid_to`` NULL). Each older member gets ``valid_to =
    winner.valid_from`` and ``superseded_by_id = winner.id`` but KEEPS
    ``is_active=True`` — so it stays recallable (the queryable-A property). This is
    the AUTOMATIC path's is_active semantics; ``mutate.supersede_memory`` (a manual
    user retire) deliberately sets ``is_active=False`` instead. No LLM, no DELETE.
    """
    winner = max(sources, key=lambda m: (m.valid_from, m.created_at, str(m.id)))
    losers = [m.id for m in sources if m.id != winner.id]
    if not losers:
        return 0
    db.query(AgentMemory).filter(AgentMemory.id.in_(losers)).update(
        {
            AgentMemory.valid_to: winner.valid_from,
            AgentMemory.superseded_by_id: winner.id,
        },
        synchronize_session=False,
    )
    return len(losers)


def insert_root_archive_sources(db, text, sources, *, supersede_children: bool = False):
    """Insert the ACTIVE synthesized root, then archive the N source leaves.

    Faithful to production ``store._insert_root_archive_sources``: identity inherits
    from the NEWEST source (candidates arrive ordered by ``(created_at, id)``),
    ``created_at``/``occurred_at`` backdated to the earliest source (preserves the
    event-reserve recency), ``valid_from = earliest_occ or earliest_created``,
    ``source_message_id=NULL`` (synthetic), union-of-tags provenance, ``hit_count=0``.
    Re-embedded via the SAME ``store._populate_search_fields`` path ``write`` uses.

    ``supersede_children`` (default False) is production's AUTOMATIC union-merge:
    children get only ``is_active=False`` + ``parent_id``. ``mutate.merge_memories``
    (a manual user merge) passes True to ALSO stamp ``valid_to`` + ``superseded_by_id``
    — the fuller user-driven audit trail. Shared so the two never fork.
    """
    newest = sources[-1]
    occ = [s.occurred_at for s in sources if s.occurred_at is not None]
    earliest_created = min(s.created_at for s in sources)
    earliest_occ = min(occ) if occ else None
    root = AgentMemory(
        id=uuid.uuid4(),
        user_id=newest.user_id,
        agent_type=newest.agent_type,
        project_id=newest.project_id,
        customer_id=newest.customer_id,
        session_id=None,
        content=text,
        context_summary=text,
        occurred_at=earliest_occ,
        created_at=earliest_created,
        valid_from=earliest_occ or earliest_created,
        hit_count=0,
        is_active=True,
        parent_id=None,
        memory_tier=newest.memory_tier,
        source_message_id=None,
    )
    PostgresMemoryStore(db)._populate_search_fields(root, tsv_source=text, embed_source=text)

    seen: set = set()
    union_tags: list[MemoryTag] = []
    for s in sources:
        for tag in s.tags:
            if tag.id not in seen:
                seen.add(tag.id)
                union_tags.append(tag)
    root.tags = union_tags

    db.add(root)
    db.flush()  # assign root.id before pointing the leaves at it

    archive_values: dict = {AgentMemory.is_active: False, AgentMemory.parent_id: root.id}
    if supersede_children:
        now = datetime.now(UTC)
        archive_values[AgentMemory.valid_to] = now
        archive_values[AgentMemory.superseded_by_id] = root.id
    db.query(AgentMemory).filter(
        AgentMemory.id.in_([s.id for s in sources])
    ).update(archive_values, synchronize_session=False)
    return root


# ── Pool pass (faithful to production store._consolidate_pool) ──────────────────

def _consolidate_pool(db, pool_filters, *, tau: float, max_groups: int) -> ConsolidationResult:
    """Shared merge pass over one candidate pool.

    Reads the pool's active, current (``valid_to IS NULL``), embedded corpus, drops
    never-merge rows, groups near-duplicates (cosine >= ``tau``), and merges each
    group. Archive-never-hard-delete: no DELETE anywhere. Per-group failure
    isolation. Commits once at the end. Convergent: re-running yields ~0 new merges
    (archived sources drop out of the active pool). ``max_groups=0`` means all.

    LLM available ⇒ faithful to production: Haiku union-synthesis, and (when
    ``MEMORY_SUPERSESSION`` is ON) LLM classification routes CONFLICT groups to
    :func:`supersede_conflict`, DISTINCT groups are skipped, RESTATEMENT groups
    union-merge. No LLM ⇒ degraded deterministic union (see module docstring).
    """
    result = ConsolidationResult()
    mems = (
        db.query(AgentMemory)
        .filter(
            *pool_filters,
            AgentMemory.is_active.is_(True),
            AgentMemory.valid_to.is_(None),
            AgentMemory.embedding.isnot(None),
        )
        .order_by(AgentMemory.created_at.asc(), AgentMemory.id.asc())
        .all()
    )
    if len(mems) < 2:
        return result  # singleton pool: skip the never-merge JOIN too
    never = never_merge_ids(db, pool_filters)
    if never:
        mems = [m for m in mems if m.id not in never]
        if len(mems) < 2:
            return result

    groups = dedup_groups([m.embedding for m in mems], tau)
    if max_groups:
        groups = groups[:max_groups]
    result.groups = len(groups)

    llm_ok = llm_available()
    supersede_on = config.MEMORY_SUPERSESSION and llm_ok
    haiku = config.resolve_model("haiku")
    for grp in groups:
        sources = [mems[i] for i in grp]
        texts = [first_nonblank(m.context_summary, m.content) for m in sources]
        if supersede_on:
            relation = classify_group(texts)
            result.cost_calls += 1  # the detection call
            if relation is GroupRelation.CONFLICT:
                result.superseded += supersede_conflict(db, sources)
                continue
            if relation is GroupRelation.DISTINCT_FACTS:
                continue  # grouped by proximity but not the same fact
            # RESTATEMENT falls through to the union-merge below.
        if llm_ok:
            try:
                text, _usage = llm_complete(
                    MERGE_PROMPT.format(facts=build_facts_block(texts)),
                    model=haiku, max_tokens=220,
                )
            except Exception:
                logger.warning(
                    "consolidation: Haiku merge failed for a group; leaving it "
                    "untouched", exc_info=True,
                )
                text = ""
            result.cost_calls += 1
            text = (text or "").strip()
            if not text:
                result.errors += 1
                continue  # faithful: never archive sources we couldn't merge
        else:
            # DEGRADED (no LLM): deterministic lossless union — the documented
            # manual/offline difference from production (which skips on no-LLM).
            text = deterministic_union_summary(sources)
            if not text:
                result.errors += 1
                continue
        insert_root_archive_sources(db, text, sources)
        result.synthesized += 1
        result.archived += len(sources)

    db.commit()
    return result


def consolidate_scope(
    db, *, user_id, agent_type: str, project_id=None,
    tau: float = DEFAULT_TAU, max_groups: int = 0,
) -> ConsolidationResult:
    """Synthesize-merge one ``(user_id, agent_type, project_id)`` pool (faithful to
    production ``store.consolidate_scope``). Mechanics in :func:`_consolidate_pool`.
    """
    return _consolidate_pool(
        db,
        [
            AgentMemory.user_id == user_id,
            AgentMemory.agent_type == agent_type,
            AgentMemory.project_id == project_id,
        ],
        tau=tau,
        max_groups=max_groups,
    )


def _active_count(db, pool_filters) -> int:
    """Count active memories in the pool (for the before/after contract fields)."""
    return db.query(AgentMemory).filter(*pool_filters, AgentMemory.is_active.is_(True)).count()


def _record_run(db, *, scope_key: str, result: ConsolidationResult, threshold: float) -> str:
    """Upsert the durable ``MemoryConsolidationRun`` ledger row for ``scope_key``.

    Mirrors production ``_record_consolidation_run`` (one upserted row per scope):
    ``last_run_at`` = timing, ``groups_merged`` / ``rows_archived`` / ``cost_calls``
    = counts. ``threshold`` is NOT a production column (the model mirrors production
    exactly), so it is LOGGED, not persisted — same precedent as ``mutate``'s
    ``reason``. Returns the ledger row id (the contract's ``run_id``).
    """
    now = datetime.now(UTC)
    ledger = (
        db.query(MemoryConsolidationRun)
        .filter(MemoryConsolidationRun.scope_key == scope_key)
        .first()
    )
    if ledger is None:
        ledger = MemoryConsolidationRun(id=uuid.uuid4(), scope_key=scope_key)
        db.add(ledger)
    ledger.last_run_at = now
    ledger.groups_merged = result.synthesized
    ledger.rows_archived = result.archived
    ledger.cost_calls = result.cost_calls
    logger.info(
        "consolidation run recorded scope=%s tau=%.3f merged=%d archived=%d superseded=%d",
        scope_key, threshold, result.synthesized, result.archived, result.superseded,
    )
    db.commit()
    return str(ledger.id)


def consolidate(persona: str, *, threshold: float = DEFAULT_TAU, max_groups: int = 0) -> dict:
    """Manual, on-demand consolidation of one persona's memory pool.

    The single public entry-point (NO background loop): resolves the persona's
    scope (``recall._scope`` → this human's ``user_id`` + the project ``project_id``,
    ``agent_type=persona``), runs the synthesize-merge pass at ``threshold`` (cosine
    near-dup τ, default 0.85), and records a ``MemoryConsolidationRun`` ledger row.

    Returns ``{run_id, pools_merged, memories_before, memories_after, superseded}``:
    ``pools_merged`` = near-dup groups merged into a parent; ``superseded`` = rows
    retired via the newest-wins conflict path; ``memories_before/after`` = active
    row counts around the pass (archived children drop ``memories_after``).
    """
    from memory.recall import _scope  # local import: avoid any import-order surprise

    uid, pid, _cid, _slug = _scope()
    pool_filters = [
        AgentMemory.user_id == uid,
        AgentMemory.agent_type == persona,
        AgentMemory.project_id == pid,
    ]
    db = get_session()
    try:
        before = _active_count(db, pool_filters)
        result = consolidate_scope(
            db, user_id=uid, agent_type=persona, project_id=pid,
            tau=threshold, max_groups=max_groups,
        )
        after = _active_count(db, pool_filters)
        run_id = _record_run(
            db, scope_key=make_scope_key(uid, persona, pid),
            result=result, threshold=threshold,
        )
        return {
            "run_id": run_id,
            "pools_merged": result.synthesized,
            "memories_before": before,
            "memories_after": after,
            "superseded": result.superseded,
        }
    finally:
        db.close()
