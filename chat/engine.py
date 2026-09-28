"""Per-turn memory hook replication — headless, dependency-light, unit-testable.

This is the CRUX of the demo: a faithful replay of the production hook loop in
``.claude/hooks/memory_hook.py`` against the replica's in-process API.

Production hook -> replica equivalent (cited):

  * PRE-hook. ``memory_hook.handle_user_prompt`` sends ``{"cmd": "recall",
    "persona", "query": prompt, "k": 24}`` to the daemon and injects the returned
    ``block`` as ``additionalContext`` (memory_hook.py:559-612). Here:
    ``backend.recall_facts(persona, user_msg, k=24)`` -> ``build_context_block`` ->
    injected into the LLM prompt.
  * REPLY. Generated with the recalled memory block in context (production runs
    the persona turn with ``additionalContext`` prepended). Here the reply is
    produced by the replica's pluggable LLM shim (``memory.llm.llm_call`` via
    ``default_llm_fn``) with the block in the prompt. No key -> the shim returns
    ``None`` -> we degrade to a recall-only view instead of crashing.
  * POST-hook. ``memory_hook.handle_stop`` sends ``{"cmd": "write", "persona",
    "transcript", "turns": 1, "mode": "auto"}`` (memory_hook.py:658-662). Here:
    ``backend.store_facts(persona, turn_text, mode="auto")`` on the finished turn.

The engine codes against the ``recall_facts`` / ``store_facts`` contract
(``eval.memory_client.MemoryBackend``), so the SAME ``run_turn`` drives the real
``EngineBackend`` and the in-memory ``StubMemoryClient`` used by the wiring test —
no DB or API key needed to prove the pre -> reply -> post order.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Preserved recall param — top-k=24, matching production (memory_hook.py:560) and
# the replica's documented parity constant (eval.memory_client.DEFAULT_K).
RECALL_K = 24
DEFAULT_PERSONA = "researcher"

# The reply's system prompt. Kept out of the injected recall block (which is the
# additionalContext), mirroring the production split of context vs. instruction.
CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory of your past conversations "
    "with this user. Use the recalled memory below when it is relevant, and do not "
    "invent facts that are not supported by it or by the user's current message. "
    "If the memory does not cover the question, say so plainly."
)

# Shown as the reply when no LLM credential is available (memory.llm.llm_call
# returned None). The recall + injected-context view still renders — a recall-only
# degrade, never a crash (task requirement; mirrors the hook's fail-soft design).
RECALL_ONLY_NOTICE = (
    "(No LLM credential set — reply generation is off, so this is a RECALL-ONLY "
    "view. The panel on the right shows the memory that WOULD be injected into the "
    "prompt. Set ANTHROPIC_API_KEY in the settings above to enable replies.)"
)

# llm_fn(prompt, system_prompt) -> reply text, or None when no LLM is available.
LlmFn = Callable[[str, str], str | None]


@dataclass
class RecalledFact:
    """One recalled memory, normalized for the debug panel — a superset of the
    fields production's recall block carries.

    ``score`` (salience) / ``matched`` / ``scope`` come from the engine's ranking
    plan and are present only against the real ``EngineBackend``; the stub leaves
    them ``None`` / ``"-"`` (a dry-run shows summary/rank/hits/date only).

    The TWO timestamps are deliberately distinct and both surfaced: ``occurred_at``
    is the EVENT time (when the fact happened) and ``created_at`` is the WRITE /
    RECENCY time (when the row was stored — what the recency-decay leg keys off).
    """

    rank: int
    summary: str
    scope: str = "-"
    score: float | None = None
    matched: bool | None = None
    hit_count: int = 0
    occurred_at: str | None = None  # EVENT time (when it happened)
    created_at: str | None = None   # WRITE / RECENCY time (when it was stored)
    by: str | None = None


@dataclass
class TurnResult:
    """Everything one turn produced — for the chat reply and the debug panel."""

    recalled: list[RecalledFact]
    context_block: str
    reply: str
    llm_used: bool
    stored: dict
    turn_text: str
    events: list[str] = field(default_factory=list)


def _to_recalled(rows: Sequence[dict]) -> list[RecalledFact]:
    """Normalize recall_facts dicts -> RecalledFact, tolerant of the poorer stub
    contract (no scope/score/matched)."""
    out: list[RecalledFact] = []
    for i, r in enumerate(rows, 1):
        out.append(
            RecalledFact(
                rank=i,
                summary=(r.get("summary") or "").strip(),
                scope=r.get("scope") or "-",
                score=r.get("score"),
                matched=r.get("matched"),
                hit_count=int(r.get("hit_count") or 0),
                occurred_at=r.get("occurred_at"),
                created_at=r.get("created_at"),
                by=r.get("by"),
            )
        )
    return out


def build_context_block(facts: Sequence[RecalledFact]) -> str:
    """Recalled facts -> the injected context block (production additionalContext).

    Numbered, most-relevant-first, with light provenance (date/scope) — the
    replica's existing convention for turning recalled facts into injected context
    (``eval.harness._build_context``), extended with the provenance the demo shows.
    """
    if not facts:
        return "(no relevant memories recalled for this message)"
    lines = ["Recalled memory (most relevant first):"]
    for f in facts:
        meta = [m for m in (f.occurred_at, f.scope if f.scope != "-" else None) if m]
        prefix = f"[{' · '.join(meta)}] " if meta else ""
        lines.append(f"{f.rank}. {prefix}{f.summary}")
    return "\n".join(lines)


def build_llm_prompt(context_block: str, user_msg: str) -> str:
    """Prepend the recalled memory block to the user turn — the injection point."""
    return f"{context_block}\n\nUser: {user_msg}"


def default_llm_fn(prompt: str, system_prompt: str, model: str | None = None) -> str | None:
    """Reply via the replica's pluggable LLM shim (``memory.llm.llm_call``).

    Reuses the engine's ONE LLM client + its env-based key mechanism
    (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ant profile). ``model`` (friendly
    name haiku/sonnet/opus or a full id; ``None`` = Haiku default) selects the
    reply model via ``memory.llm.llm_call``. Returns None when no credential/SDK is
    available — never raises. Imported lazily so the wiring test imports this module
    without the ``anthropic`` SDK installed.
    """
    from memory.llm import llm_call

    return llm_call(prompt, system_prompt=system_prompt, model=model, max_tokens=512)


def run_turn(
    backend,
    persona: str,
    user_msg: str,
    *,
    k: int = RECALL_K,
    llm_fn: LlmFn | None = None,
    model: str | None = None,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
) -> TurnResult:
    """Replay one production turn: PRE-hook recall -> reply -> POST-hook write.

    ``backend`` implements the ``recall_facts`` / ``store_facts`` contract
    (``EngineBackend`` for the real engine, ``StubMemoryClient`` for tests).
    ``events`` records the actual call order so a test can assert PRE precedes the
    reply precedes POST.

    ``model`` (friendly name haiku/sonnet/opus or a full id) selects the LLM for
    BOTH sides of the turn — the reply (default ``llm_fn`` -> ``memory.llm.llm_call``)
    and the POST-hook store (``store_facts`` -> extraction). It is threaded to
    ``store_facts`` only when set, so the leaner stub/test backends (whose
    ``store_facts`` takes no ``model``) keep working when ``model`` is ``None``.
    """
    persona = (persona or DEFAULT_PERSONA).strip() or DEFAULT_PERSONA
    user_msg = (user_msg or "").strip()
    # Bind the chosen model into the default reply fn; an injected llm_fn is used
    # as-is (the wiring test supplies its own 2-arg fake).
    llm_fn = llm_fn or (lambda prompt, sys_prompt: default_llm_fn(prompt, sys_prompt, model=model))
    events: list[str] = []

    # ── PRE-hook (UserPromptSubmit -> recall + inject) ──────────────────────
    rows = backend.recall_facts(persona, user_msg, k=k)
    events.append("recall")
    facts = _to_recalled(rows)
    context_block = build_context_block(facts)

    # ── Reply, generated WITH the recalled memory injected into the prompt ──
    prompt = build_llm_prompt(context_block, user_msg)
    reply_text = llm_fn(prompt, system_prompt)
    events.append("llm")
    llm_used = reply_text is not None
    reply = reply_text if llm_used else RECALL_ONLY_NOTICE

    # ── POST-hook (Stop -> extract + store the finished turn) ───────────────
    # Turn framing matches the extraction prompt's "User:"/"Agent:" contract
    # (memory.extraction) so the gated/atomic extractor grounds on the user's words.
    turn_text = f"User: {user_msg}\nAgent: {reply}"
    store_kwargs = {"model": model} if model else {}
    stored = backend.store_facts(persona, turn_text, mode="auto", **store_kwargs)
    events.append("store")

    return TurnResult(
        recalled=facts,
        context_block=context_block,
        reply=reply,
        llm_used=llm_used,
        stored=stored,
        turn_text=turn_text,
        events=events,
    )


class EngineBackend:
    """Real wiring to the memory-research engine.

    ``recall_facts`` returns a SUPERSET of the public contract: alongside the
    documented ``summary``/``occurred_at``/``hit_count``/``by`` it carries
    ``scope`` (own vs. team) plus the ranking plan's ``score`` (salience) and
    ``matched`` (relevance-matched vs. recency-reserve) — the fields the debug
    panel needs. It computes them by replaying ``memory.recall.recall_facts``'s own
    body verbatim (``store.recall_project_sections`` then ``record_recall_hits``),
    so ranking + the single hit-writeback stay byte-faithful; it only keeps the
    section/plan data that the public wrapper collapses. ``store_facts`` delegates
    unchanged. All engine imports are lazy so importing ``chat.engine`` (for the
    stub-backed test) needs neither the DB nor the ONNX/embedding stack.
    """

    def recall_facts(self, persona: str, query: str, k: int = RECALL_K) -> list[dict]:
        # Reuse the engine's own scope + contributor helpers to stay faithful to
        # recall_facts rather than re-deriving the scope UUIDs / by:<tag> join.
        from memory.db import get_session
        from memory.recall import _contributors, _scope
        from memory.store import PostgresMemoryStore

        uid, pid, cid, _slug = _scope()
        db = get_session()
        try:
            store = PostgresMemoryStore(db)
            own, own_plan, team, team_plan = store.recall_project_sections(
                user_id=uid, agent_type=persona, project_id=pid,
                customer_id=cid, query=query, k=k,
            )
            if own_plan or team_plan:
                store.record_recall_hits(
                    own_plan + team_plan, agent_type=persona, refresh_instances=True
                )
            # plan rows are (memory_id, salience, relevance_matched).
            plan_by_id = {mid: (sal, matched) for mid, sal, matched in own_plan + team_plan}
            rows: list[dict] = []
            for scope_label, mems in (("own", own), ("team", team)):
                for m in mems:
                    sal_matched = plan_by_id.get(m.id)
                    rows.append({
                        "summary": (m.context_summary or m.content or "").strip(),
                        # EVENT time (when it happened) vs WRITE/RECENCY time (when
                        # it was stored) — both surfaced so the panel can distinguish
                        # them, matching production's recall block.
                        "occurred_at": m.occurred_at.date().isoformat() if m.occurred_at else None,
                        "created_at": m.created_at.date().isoformat() if m.created_at else None,
                        "hit_count": m.hit_count,
                        "by": _contributors(m) or None,
                        "scope": scope_label,
                        "score": sal_matched[0] if sal_matched else None,
                        "matched": sal_matched[1] if sal_matched else None,
                    })
            return rows
        finally:
            db.close()

    def store_facts(self, persona: str, content: str, mode: str = "auto",
                    model: str | None = None) -> dict:
        from memory.recall import store_facts

        return store_facts(persona, content, mode=mode, model=model)
