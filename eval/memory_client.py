"""Thin wrapper over the memory engine's public API.

The engine (built in parallel — do NOT depend on its branch) exposes, per
Nizam's interface contract::

    memory.recall.recall_facts(persona, query, k=24)
        -> list[dict{summary, occurred_at, hit_count, by}]
    memory.recall.store_facts(persona, content, mode='auto')       -> dict
    memory.recall.store_facts_verbatim(persona, facts)             -> dict

Recall params: top-k=24, RRF K=60, half-life 7d. The RRF leg weights, the
per-leg candidate cap and the lexical-OR switch are env-overridable in
``memory.config``; ``live_recall_params()`` reads the values actually in effect
so the results JSON records them.

Everything goes through this one file so that if the import path shifts at
integration, only ``_ENGINE_IMPORT_PATH`` changes. The import is lazy — a
harness dry-run uses ``StubMemoryClient`` and never touches the engine.
"""
from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

_ENGINE_IMPORT_PATH = "memory.recall"

# Recall parameter DEFAULTS (memory/config.py with no env overrides). For what
# a run actually used, call live_recall_params(); these are display fallbacks.
DEFAULT_K = 24
RRF_K = 60
RRF_WEIGHTS = (1.0, 1.0, 2.0)  # fuzzy / tag / vector
HALF_LIFE_DAYS = 7


def live_recall_params() -> dict:
    """The engine's recall parameters as currently configured, env overrides
    included (MEMORY_RRF_W_FUZZY / _TAG / _VECTOR, MEMORY_CANDIDATE_LIMIT,
    MEMORY_LEXICAL_OR). ``memory.config`` reads the environment at import, so
    this reflects the process the run executes in."""
    cfg = importlib.import_module("memory.config")
    return {
        "rrf_k": cfg.MEMORY_RRF_K,
        "weights": [cfg.MEMORY_RRF_W_FUZZY, cfg.MEMORY_RRF_W_TAG, cfg.MEMORY_RRF_W_VECTOR],
        "weights_order": ["fuzzy", "tag", "vector"],
        "candidate_limit": cfg.CANDIDATE_LIMIT,
        "lexical_or": cfg.MEMORY_LEXICAL_OR,
        "half_life_days": cfg.RECENCY_HALF_LIFE_DAYS,
    }


class MemoryBackend(Protocol):
    """The surface the harness codes against (real engine or stub)."""

    def store_facts_verbatim(self, persona: str, facts: Sequence[str]) -> dict: ...
    def store_facts(self, persona: str, content: str, mode: str = "auto") -> dict: ...
    def recall_facts(
        self, persona: str, query: str, k: int = DEFAULT_K, update_hits: bool = True
    ) -> list[dict]: ...
    def count_facts(self, persona: str) -> int: ...


class MemoryClient:
    """Adapter over the real engine module. Imported lazily on first use."""

    def __init__(self, import_path: str = _ENGINE_IMPORT_PATH):
        self._import_path = import_path
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            self._engine = importlib.import_module(self._import_path)
        return self._engine

    def store_facts_verbatim(self, persona: str, facts: Sequence[str]) -> dict:
        return self._load().store_facts_verbatim(persona, list(facts))

    def store_facts(self, persona: str, content: str, mode: str = "auto") -> dict:
        return self._load().store_facts(persona, content, mode=mode)

    def recall_facts(
        self, persona: str, query: str, k: int = DEFAULT_K, update_hits: bool = True
    ) -> list[dict]:
        """``update_hits=False`` is a read-only recall — no hit-count/salience
        mutation — so an eval that snapshots recall twice (e.g. the consolidation
        BEFORE/AFTER sweep) does not perturb ranking between the two passes."""
        return self._load().recall_facts(persona, query, k=k, update_hits=update_hits)

    def count_facts(self, persona: str) -> int:
        """Cheap scope-filtered existence COUNT — the reuse-ingest detector.

        Counts active rows in the EXACT own-scope ``store_facts_verbatim`` writes to
        and ``recall_facts`` reads from — ``(user_id, agent_type=persona, project_id,
        is_active)`` — so a non-zero count deterministically means recall for this
        persona will see prior memories (it does NOT depend on any query text). Reuses
        the engine's single-sourced ``_scope`` / ``get_session`` / ``AgentMemory``
        (fetched through the one lazy import path, so a stub dry-run never imports the
        engine) rather than re-deriving scope, so it can never drift from the
        write/read scope.
        """
        mod = self._load()
        uid, pid, _cid, _slug = mod._scope()
        db = mod.get_session()
        try:
            return (
                db.query(mod.AgentMemory.id)
                .filter(
                    mod.AgentMemory.user_id == uid,
                    mod.AgentMemory.agent_type == persona,
                    mod.AgentMemory.project_id == pid,
                    mod.AgentMemory.is_active.is_(True),
                )
                .count()
            )
        finally:
            db.close()


_WORD = re.compile(r"[a-z0-9]+")


class StubMemoryClient:
    """In-memory, dependency-free backend for testing / dry-runs.

    Stores facts verbatim per persona and ranks recall by deterministic
    token-overlap (Jaccard), highest first — enough to exercise the full
    harness (ingest → recall → score) without the engine or an API key. The
    returned dict shape matches the engine contract: summary/occurred_at/
    hit_count/by.
    """

    def __init__(self):
        self._store: dict[str, list[str]] = {}

    def store_facts_verbatim(self, persona: str, facts: Sequence[str]) -> dict:
        bucket = self._store.setdefault(persona, [])
        bucket.extend(facts)
        return {"stored": len(facts)}

    def store_facts(self, persona: str, content: str, mode: str = "auto") -> dict:
        return self.store_facts_verbatim(persona, [content])

    def count_facts(self, persona: str) -> int:
        """Number of facts stored for ``persona`` — the stub's reuse-ingest detector
        (mirrors the engine's scope-filtered COUNT: >0 means recall will see prior
        memories for this persona)."""
        return len(self._store.get(persona, []))

    def recall_facts(
        self, persona: str, query: str, k: int = DEFAULT_K, update_hits: bool = True
    ) -> list[dict]:
        # update_hits is accepted for protocol parity; the stub keeps no hit state.
        q = Counter(_WORD.findall(query.lower()))
        scored: list[tuple[float, str]] = []
        for fact in self._store.get(persona, []):
            f = Counter(_WORD.findall(fact.lower()))
            inter = sum((q & f).values())
            union = sum((q | f).values()) or 1
            scored.append((inter / union, fact))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"summary": fact, "occurred_at": None, "hit_count": 0, "by": persona}
            for score, fact in scored[:k]
            if score > 0
        ]
