"""PostgresMemoryStore — the recall + write core, ported from production ``store.py``.

Write path: one durable row per memory + canonical-tag reconciliation + a MiniLM-384
embedding (when on PostgreSQL with the embedder available).

Read path (``recall``): scope filter FIRST — ``(user_id, agent_type, project_id,
is_active)``, with ``session_id`` deliberately OMITTED so cross-session recall works.
Candidates compose four retrieval methods (§5): structured (recency), fuzzy
(``websearch_to_tsquery`` graded by ``ts_rank_cd``; LIKE fallback on SQLite),
entity-tag (per-scope registry), and vector-ANN (cosine ``embedding <=> :qvec``). The
fuzzy/tag/vector lists are fused by weighted RRF (``_rrf_fuse``); ``recent`` is the
fallback pool. Ranking priority is MATCH -> HITS -> RECENCY; salience encodes the same
ordering as one inspectable float.

Trimmed vs production: the synthesize-merge consolidation write path and the
daemon's deferred-plan machinery are removed. The recall math, RRF, salience,
event-fresh reserve, and hit writeback are byte-faithful. Flagsmith flags and the
pydantic ``Settings`` object are replaced by ``memory.config`` constants (all flags ON).
"""
from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime

from sqlalchemy import case, func, or_, update
from sqlalchemy.orm import Session, defer, lazyload, selectinload

from memory import config
from memory.embeddings import (
    EMBEDDER_UNAVAILABLE_HINT,
    first_nonblank,
    get_embedding_service,
)
from memory.extraction import normalize_label, reconcile_tags
from memory.models import AgentMemory, MemoryTag, MemoryTagLink

logger = logging.getLogger(__name__)

# ── Ranking constants (memory_design.md §5) — mirror memory.config ─────────────
RECENCY_HALF_LIFE_DAYS = config.RECENCY_HALF_LIFE_DAYS
_CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
_RRF_K = config.MEMORY_RRF_K
_SALIENCE_MATCH_BAND = config.SALIENCE_MATCH_BAND
_TS_CONFIG = "english"
_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "what", "how", "did", "was", "were",
     "you", "about", "tell", "this", "that", "are", "can"}
)


def make_scope_key(user_id, agent_type: str, project_id=None) -> str:
    """Canonical tag-registry scope key: ``user|agent|project-or-global``."""
    return f"{user_id}|{agent_type}|{project_id or 'global'}"


def _with_time_bounds(
    scope: list, occurred_after: datetime | None, occurred_before: datetime | None
) -> list:
    """Append the §5 strictly-temporal EVENT-time filter to a scope list in place."""
    if occurred_after is not None:
        scope.append(AgentMemory.occurred_at >= occurred_after)
    if occurred_before is not None:
        scope.append(AgentMemory.occurred_at <= occurred_before)
    return scope


def _query_terms(query: str) -> list[str]:
    """Lowercase alphanumeric terms (len >= 3, minus stopwords), max 8."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS][:8]


def _term_overlap(terms: list[str], *texts: str | None) -> int:
    """#22 graded proxy for SQLite: distinct query terms present in the text."""
    haystack = " ".join(t.lower() for t in texts if t)
    return sum(1 for term in terms if term in haystack)


def _rrf_fuse(ranked_lists, k: int = _RRF_K, weights=None) -> dict:
    """Weighted Reciprocal Rank Fusion (weights tunable) — PURE.

    Returns ``{id: rrf_score}`` where the score is ``Σ_lists weight_list · 1/(k +
    rank)`` (rank 0-based). A ``0.0`` weight drops that leg entirely (true no-op).
    With default weights and only the lexical+tag lists non-empty, the fused order
    matches the pre-vector relevance signal exactly.
    """
    scores: dict = {}
    for i, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else (weights[i] if i < len(weights) else 1.0)
        if weight == 0.0:
            continue
        for rank, mem_id in enumerate(ranked):
            scores[mem_id] = scores.get(mem_id, 0.0) + weight / (k + rank)
    return scores


def _aware(dt: datetime) -> datetime:
    """Treat a naive timestamp (SQLite) as UTC; pass tz-aware ones through."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


# ── Degraded-pipeline warnings ────────────────────────────────────────────────
# The vector leg carries RRF weight 4.0 of 6.0 (§A5) — the DOMINANT relevance
# signal. On a non-PostgreSQL backend it silently returns [] and recall still
# "works", ranking on the lexical+tag legs alone: plausible-looking numbers that
# are NOT comparable to the production baseline. The write path skips the
# embedding just as silently, so the corpus itself is left unusable. Both are
# warned LOUD.
#
# Latched once per reason per process: recall is a hot path (an eval sweep issues
# thousands of calls), so an unlatched log would be unreadable spam — but a silent
# skip is exactly how a fully-disabled vector leg survives a whole study
# unnoticed. Once-per-reason is the compromise. WARNING reaches stderr through
# ``logging.lastResort`` even when no consumer has configured logging.
_DEGRADED_HINT = (
    "Fix: point MEMORY_DATABASE_URL at a PostgreSQL + pgvector database "
    "(see docker-compose.yml / run_demo.sh)."
)

_warned_once: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Emit a WARNING the first time ``key`` is seen in this process."""
    if key in _warned_once:
        return
    _warned_once.add(key)
    logger.warning(message, *args)


def reset_degraded_warnings() -> None:
    """Clear the once-per-process warning latch. **Test helper only.**"""
    _warned_once.clear()


def _rrf_weight_total() -> float:
    """Total RRF weight across the three legs — read live so a reloaded config
    (tests, env overrides) can never leave a stale figure in a warning."""
    return (
        config.MEMORY_RRF_W_FUZZY + config.MEMORY_RRF_W_TAG + config.MEMORY_RRF_W_VECTOR
    )


class PostgresMemoryStore:
    """Durable Postgres-backed working-memory store (SQLite-degradable)."""

    def __init__(self, db: Session):
        self.db = db

    @property
    def _is_postgres(self) -> bool:
        return self.db.get_bind().dialect.name == "postgresql"

    # ── write path ────────────────────────────────────────────────────────

    def write(
        self,
        *,
        user_id,
        agent_type: str,
        project_id=None,
        customer_id=None,
        session_id=None,
        content: str,
        context_summary: str | None = None,
        agent_summary: str | None = None,
        tags: list[str] | None = None,
        discipline: str | None = None,
        authority: str | None = None,
        occurred_at: datetime | None = None,
        source_message_id=None,
        idempotency_key: str | None = None,
    ) -> AgentMemory:
        """Insert one memory + reconcile/link tags. Idempotent on source_message_id.

        ``agent_summary`` (two-section cell) is recall payload only — DELIBERATELY
        excluded from both ``content_tsv`` and the embedding input. With an
        ``idempotency_key`` given, dedup keys on ``(source_message_id,
        context_summary)`` so a turn's N atomic facts each get their own identity.
        Commits the session.
        """
        if source_message_id is not None:
            idem_q = self.db.query(AgentMemory).filter(
                AgentMemory.source_message_id == source_message_id
            )
            if idempotency_key is not None:
                idem_q = idem_q.filter(AgentMemory.context_summary == context_summary)
            existing = idem_q.first()
            if existing is not None:
                logger.debug(
                    "memory write: idempotent hit for source_message_id=%s (key=%s)",
                    source_message_id, idempotency_key,
                )
                return existing

        memory = AgentMemory(
            user_id=user_id,
            agent_type=agent_type,
            project_id=project_id,
            customer_id=customer_id,
            session_id=session_id,
            content=content,
            context_summary=context_summary,
            agent_summary=agent_summary,
            discipline=discipline,
            authority=authority,
            occurred_at=occurred_at,
            source_message_id=source_message_id,
        )
        if occurred_at is not None:
            memory.valid_from = occurred_at

        tsv_source = content if not context_summary else f"{content}\n{context_summary}"
        self._populate_search_fields(
            memory,
            tsv_source=tsv_source,
            embed_source=first_nonblank(context_summary, content),
        )

        if tags:
            scope_key = make_scope_key(user_id, agent_type, project_id)
            memory.tags = reconcile_tags(self.db, scope_key, tags)

        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        return memory

    def _populate_search_fields(self, row, *, tsv_source: str, embed_source: str | None) -> None:
        """PG-only: set ``content_tsv`` + ``embedding`` (no-op on SQLite). A NULL
        vector is logged LOUD + repairable, never a silent commit (§8).

        The non-PostgreSQL skip warns once too: it is the WRITE-side half of the
        same defect as the disabled vector leg, and the more damaging half — the
        rows it produces carry no embedding at all, so they stay unusable by the
        vector leg even after the backend is fixed.
        """
        if not self._is_postgres:
            _warn_once(
                "write_no_search_fields",
                "MEMORY WRITE DEGRADED — database dialect is %r, not 'postgresql': "
                "rows are being stored with NO embedding and NO content_tsv. Recall "
                "cannot use the vector or full-text legs over this corpus, and these "
                "rows must be RE-INGESTED (or their embeddings backfilled) once a "
                "pgvector database is in use — repointing the DSN alone will not "
                "repair them. %s",
                self.db.get_bind().dialect.name,
                _DEGRADED_HINT,
            )
            return
        row.content_tsv = func.to_tsvector(_TS_CONFIG, tsv_source)
        vector = get_embedding_service("memory").embed(embed_source) if embed_source else None
        if vector is not None:
            row.embedding = vector
        else:
            logger.error(
                "Memory embedding unavailable — row stored with a NULL vector "
                "(repairable). %s",
                EMBEDDER_UNAVAILABLE_HINT,
            )

    # ── read path ─────────────────────────────────────────────────────────

    def recall(
        self,
        *,
        user_id,
        agent_type: str,
        project_id=None,
        query: str,
        k: int = 8,
        update_hits: bool = True,
        occurred_before: datetime | None = None,
        occurred_after: datetime | None = None,
    ) -> list[AgentMemory]:
        """Scope-filtered, salience-ranked recall (§5). Thin wrapper over
        ``recall_with_plan``; applies the batched hit writeback when
        ``update_hits`` is True. Only a RELEVANCE-matched recall (rrf > 0) is a hit.
        """
        rows, plan = self.recall_with_plan(
            user_id=user_id,
            agent_type=agent_type,
            project_id=project_id,
            query=query,
            k=k,
            occurred_before=occurred_before,
            occurred_after=occurred_after,
        )
        if update_hits and plan:
            self.record_recall_hits(plan, agent_type=agent_type, refresh_instances=True)
        return rows

    def recall_with_plan(
        self,
        *,
        user_id,
        agent_type: str,
        project_id=None,
        query: str,
        k: int = 8,
        occurred_before: datetime | None = None,
        occurred_after: datetime | None = None,
    ) -> tuple[list[AgentMemory], list[tuple]]:
        """Rank WITHOUT any writeback; return ``(rows, hit_plan)``. Own-scope call."""
        scope = _with_time_bounds(
            [
                AgentMemory.user_id == user_id,
                AgentMemory.agent_type == agent_type,
                AgentMemory.project_id == project_id,  # None -> IS NULL
                AgentMemory.is_active.is_(True),
            ],
            occurred_after,
            occurred_before,
        )
        return self._recall_scoped(
            scope=scope,
            query=query,
            k=k,
            tag_scope_key=make_scope_key(user_id, agent_type, project_id),
        )

    def _recall_scoped(
        self, *, scope: list, query: str, k: int, tag_scope_key: str | None
    ) -> tuple[list[AgentMemory], list[tuple]]:
        """Shared retrieval core (structured/fuzzy/tag/vector -> RRF -> recency
        reserve -> top-k). Isolation comes ENTIRELY from ``scope``.

        LOADER OPTIONS (performance only -- they change WHAT IS FETCHED, never
        what is ranked; ``ranking_invariance.py`` asserts a byte-identical
        top-50 id list over 300 sampled LoCoMo questions):

        * ``defer(embedding)``. Each of the four legs pulls up to
          ``_CANDIDATE_LIMIT`` rows, so a query hydrates ~4x200 ORM instances.
          Nothing between here and ``_materialize`` ever READS ``.embedding`` --
          the vector leg orders by ``embedding <=> :qvec`` inside PostgreSQL, on
          the column, not on the attribute. Loading it anyway shipped and parsed
          800 vectors per query out of the wire format; at 1536-d that measured
          0.633 s of the 1.14 s recall (55%), and it is pure waste. Deferring it
          leaves the attribute loadable on demand, so any caller that does want
          it still gets it (one extra SELECT), and the write path is untouched.
        * ``lazyload(tags)`` + one ``selectinload`` over the TOP-K only. The
          mapper declares ``lazy="selectin"``, so every leg's ``.all()`` fired a
          second statement to hydrate the tags of all 200 candidates -- three
          such statements per query, 0.14 s -- when the only rows whose tags are
          ever read are the <=k that ``_materialize`` renders (``by:<slug>``
          authorship). The batched reload below fetches exactly those.
        """
        base = (
            self.db.query(AgentMemory)
            .options(defer(AgentMemory.embedding), lazyload(AgentMemory.tags))
            .filter(*scope)
        )

        # 1) structured — recency-ordered scope slice
        recent = base.order_by(AgentMemory.created_at.desc()).limit(_CANDIDATE_LIMIT).all()

        # 2) fuzzy / lexical — GRADED by relevance (#22)
        fuzzy: list[AgentMemory] = []
        if query and query.strip():
            if self._is_postgres:
                # websearch_to_tsquery ANDs every term, so a natural-language
                # question ("When did Caroline go to the LGBTQ support group?")
                # becomes 'carolin & go & lgbtq & support & group' and must match
                # entirely inside one ~30-word row. Measured over 60 LoCoMo
                # questions: 70% returned ZERO rows, mean 1.10 rows -- the
                # lexical leg was effectively dead and retrieval was a pure
                # vector kNN. OR-ing the same stopword-stripped terms
                # _query_terms() already computes for the SQLite branch returns
                # a mean of 319.7 rows with 0% zero-hit, and ts_rank_cd still
                # grades them by cover density, so precision comes from the rank
                # rather than from an all-or-nothing filter.
                _terms = _query_terms(query) if config.MEMORY_LEXICAL_OR else []
                if _terms:
                    ts_query = func.to_tsquery(
                        _TS_CONFIG, " | ".join(re.sub(r"[^a-z0-9]", "", t) for t in _terms)
                    )
                else:                       # nothing usable survived stripping
                    ts_query = func.websearch_to_tsquery(_TS_CONFIG, query)
                rank = func.ts_rank_cd(AgentMemory.content_tsv, ts_query)
                fuzzy = (
                    base.filter(AgentMemory.content_tsv.isnot(None))
                    .filter(AgentMemory.content_tsv.op("@@")(ts_query))
                    .order_by(rank.desc(), AgentMemory.created_at.desc())
                    .limit(_CANDIDATE_LIMIT)
                    .all()
                )
            else:
                terms = _query_terms(query)
                if terms:
                    clauses = []
                    for term in terms:
                        pattern = f"%{term}%"
                        clauses.append(AgentMemory.content.ilike(pattern))
                        clauses.append(AgentMemory.context_summary.ilike(pattern))
                    matched = (
                        base.filter(or_(*clauses))
                        .order_by(AgentMemory.created_at.desc())
                        .all()
                    )
                    matched.sort(
                        key=lambda m: _term_overlap(terms, m.content, m.context_summary),
                        reverse=True,
                    )
                    fuzzy = matched[:_CANDIDATE_LIMIT]

        # 3) entity-tag — normalized query terms vs. the per-scope registry
        tag_matched: list[AgentMemory] = []
        labels = {normalize_label(t) for t in _query_terms(query or "")}
        full = normalize_label(query or "")
        if full:
            labels.add(full)
        labels.discard("")
        if labels:
            tag_query = base.join(
                MemoryTagLink, MemoryTagLink.memory_id == AgentMemory.id
            ).join(MemoryTag, MemoryTag.id == MemoryTagLink.tag_id)
            if tag_scope_key is not None:
                tag_query = tag_query.filter(MemoryTag.scope_key == tag_scope_key)
            tag_matched = (
                tag_query.filter(MemoryTag.label.in_(sorted(labels)))
                .limit(_CANDIDATE_LIMIT)
                .all()
            )

        # 4) vector-ANN — PG-only, gated by the read flag AND a loadable embedder
        vector_matched = self._vector_candidates(base, query)

        # Per-method ranked id lists -> weighted RRF fusion.
        rrf_scores = _rrf_fuse(
            (
                [m.id for m in fuzzy],
                [m.id for m in tag_matched],
                [m.id for m in vector_matched],
            ),
            k=config.MEMORY_RRF_K,
            weights=(
                config.MEMORY_RRF_W_FUZZY,
                config.MEMORY_RRF_W_TAG,
                config.MEMORY_RRF_W_VECTOR,
            ),
        )

        # `recent` is the candidate POOL/fallback only (never a relevance leg).
        candidates: dict = {}
        for mem in (*recent, *fuzzy, *tag_matched, *vector_matched):
            candidates[mem.id] = mem

        now = datetime.now(UTC)
        scored: list[tuple[tuple, float, AgentMemory]] = []
        for mem in candidates.values():
            created = mem.created_at
            if created.tzinfo is None:  # SQLite returns naive UTC
                created = created.replace(tzinfo=UTC)
            age_days = max((now - created).total_seconds() / 86400.0, 0.0)
            recency_decay = 2.0 ** (-age_days / RECENCY_HALF_LIFE_DAYS)
            relevance = rrf_scores.get(mem.id, 0.0)
            hits = math.log1p(mem.hit_count or 0)
            # Priority: relevance(RRF) -> hits -> recency.
            rank_key = (relevance, hits, recency_decay)
            matched_band = _SALIENCE_MATCH_BAND if relevance > 0.0 else 0.0
            salience = matched_band + relevance * 100.0 + hits * 10.0 + recency_decay
            scored.append((rank_key, salience, mem))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Reserve a few k-slots for the FRESHEST in-scope facts (anti-feedback-loop).
        k = max(k, 0)
        reserve = min(config.MEMORY_RECALL_RECENT_RESERVE, k)
        top = scored[: k - reserve] if reserve else scored[:k]
        if reserve and len(top) < len(scored):
            chosen = {mem.id for _, _, mem in top}
            leftovers = [t for t in scored if t[2].id not in chosen]
            if config.MEMORY_EVENT_RESERVE:
                pool = [t for t in leftovers if t[0][0] > 0.0]
                pool.sort(
                    key=lambda t: _aware(t[2].occurred_at or t[2].created_at), reverse=True
                )
                if len(pool) < reserve:
                    rest = [t for t in leftovers if t[0][0] <= 0.0]
                    rest.sort(key=lambda t: _aware(t[2].created_at), reverse=True)
                    pool = pool + rest
            else:
                pool = leftovers
                pool.sort(key=lambda t: _aware(t[2].created_at), reverse=True)
            top = top + pool[:reserve]
            if len(top) < k:
                have = {mem.id for _, _, mem in top}
                top = top + [t for t in scored if t[2].id not in have][: k - len(top)]

        # Tags for the RETURNED rows only (see the loader note above). One
        # batched statement, issued after the ranking is already decided, so it
        # cannot influence it. The instances are already in this Session's
        # identity map, so this populates ``.tags`` on the very objects returned.
        top_ids = [mem.id for _, _, mem in top]
        if top_ids:
            (
                self.db.query(AgentMemory)
                .options(defer(AgentMemory.embedding), selectinload(AgentMemory.tags))
                .filter(AgentMemory.id.in_(top_ids))
                .all()
            )

        plan = [(mem.id, salience, rank_key[0] > 0.0) for rank_key, salience, mem in top]
        return [mem for _, _, mem in top], plan

    def recall_project_sections(
        self,
        *,
        user_id,
        agent_type: str,
        project_id,
        customer_id=None,
        query: str,
        k: int = 8,
        occurred_before: datetime | None = None,
        occurred_after: datetime | None = None,
    ) -> tuple[list[AgentMemory], list[tuple], list[AgentMemory], list[tuple]]:
        """Project Memory: two-section recall, READ-ONLY.

        Returns ``(own_rows, own_plan, team_rows, team_plan)``. OWN is
        ``recall_with_plan``'s scope. TEAM is a SECOND scope pass over the SAME
        table: same customer_id AND project_id AND NOT (user_id AND agent_type) AND
        is_active — everything active in the project EXCEPT the caller's own scope,
        so the two sections are disjoint by construction. When project_id/customer_id
        is unavailable (or the flag is off) the team pass is skipped and this is
        byte-identical to ``recall_with_plan``. Flex split: TEAM caps at
        ``MEMORY_PROJECT_RECALL_K``; OWN backfills to ``k - len(team_rows)``.
        """
        team_rows: list[AgentMemory] = []
        team_plan: list[tuple] = []
        own_k = k
        if config.MEMORY_PROJECT_ENABLED and project_id is not None and customer_id is not None:
            team_cap = min(config.MEMORY_PROJECT_RECALL_K, max(k, 0))
            team_scope = _with_time_bounds(
                [
                    AgentMemory.customer_id == customer_id,
                    AgentMemory.project_id == project_id,
                    or_(
                        AgentMemory.user_id != user_id,
                        AgentMemory.agent_type != agent_type,
                    ),
                    AgentMemory.is_active.is_(True),
                ],
                occurred_after,
                occurred_before,
            )
            team_rows, team_plan = self._recall_scoped(
                scope=team_scope, query=query, k=team_cap, tag_scope_key=None
            )
            own_k = (
                max(k - team_cap, 0)
                if config.MEMORY_PROJECT_RECALL_STRICT_SPLIT
                else k - len(team_rows)
            )
        own_rows, own_plan = self.recall_with_plan(
            user_id=user_id,
            agent_type=agent_type,
            project_id=project_id,
            query=query,
            k=own_k,
            occurred_before=occurred_before,
            occurred_after=occurred_after,
        )
        return own_rows, own_plan, team_rows, team_plan

    def record_recall_hits(
        self,
        plan: list[tuple],
        *,
        agent_type: str,
        now: datetime | None = None,
        refresh_instances: bool = False,
    ) -> int:
        """Apply the recall hit/usage writeback as ONE batched UPDATE + commit.

        ``plan`` rows are ``(memory_id, salience, relevance_matched)``. Every row
        gets ``last_used_at``/``salience``; ONLY relevance-matched rows accrue
        ``hit_count``/``last_hit_by``. ``refresh_instances`` re-SELECTs updated rows.
        """
        if not plan:
            return 0
        now = now or datetime.now(UTC)
        salience_by_id = {mem_id: sal for mem_id, sal, _ in plan}
        hit_ids = [mem_id for mem_id, _, matched in plan if matched]
        values: dict = {
            "last_used_at": now,
            "salience": case(
                *[(AgentMemory.id == mem_id, sal) for mem_id, sal in salience_by_id.items()]
            ),
        }
        if hit_ids:
            values["hit_count"] = case(
                (AgentMemory.id.in_(hit_ids), func.coalesce(AgentMemory.hit_count, 0) + 1),
                else_=AgentMemory.hit_count,
            )
            values["last_hit_by"] = case(
                (AgentMemory.id.in_(hit_ids), agent_type),
                else_=AgentMemory.last_hit_by,
            )
        self.db.execute(
            update(AgentMemory)
            .where(AgentMemory.id.in_(list(salience_by_id)))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        self.db.commit()
        if refresh_instances:
            self.db.query(AgentMemory).filter(
                AgentMemory.id.in_(list(salience_by_id))
            ).all()
        return len(plan)

    def _vector_candidates(self, base, query: str) -> list[AgentMemory]:
        """Vector-ANN candidate leg — empty unless PG + flag ON + loadable embedder.

        Each DISABLED path warns once (``_warn_once``); a blank query does not,
        being a legitimately empty leg rather than a degradation.
        """
        if not query or not query.strip():
            return []
        if not self._is_postgres:
            _warn_once(
                "vector_leg_not_postgres",
                "MEMORY VECTOR LEG DISABLED — database dialect is %r, not "
                "'postgresql', so pgvector is unavailable. Recall is ranking on the "
                "lexical+tag legs ONLY (the vector leg carries RRF weight %.1f of "
                "%.1f). Results are NOT comparable to the production baseline. %s",
                self.db.get_bind().dialect.name,
                config.MEMORY_RRF_W_VECTOR,
                _rrf_weight_total(),
                _DEGRADED_HINT,
            )
            return []
        if not config.MEMORY_EMBEDDINGS_ENABLED:
            _warn_once(
                "vector_leg_flag_off",
                "MEMORY VECTOR LEG DISABLED — MEMORY_EMBEDDINGS_ENABLED is OFF, so "
                "recall is ranking on the lexical+tag legs ONLY (the vector leg "
                "carries RRF weight %.1f of %.1f). Intended only as a deliberate A/B "
                "control; unset the env var to restore the production default (ON).",
                config.MEMORY_RRF_W_VECTOR,
                _rrf_weight_total(),
            )
            return []
        qvec = get_embedding_service("memory").embed(query, intent="query")
        if qvec is None:
            logger.error(
                "Memory recall vector leg unavailable — degraded to lexical+tag. %s",
                EMBEDDER_UNAVAILABLE_HINT,
            )
            return []
        return (
            base.filter(AgentMemory.embedding.isnot(None))
            .order_by(AgentMemory.embedding.cosine_distance(qvec))
            .limit(_CANDIDATE_LIMIT)
            .all()
        )
