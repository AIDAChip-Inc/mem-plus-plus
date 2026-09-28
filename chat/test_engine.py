"""Headless wiring test: proves the PRE -> reply -> POST hook order fires against
a stub memory client — no Postgres, no embedding stack, no API key.

Reuses ``eval.memory_client.StubMemoryClient`` (the in-memory backend already
shipped for dry-runs) rather than hand-rolling a second stub. Order is asserted on
``TurnResult.events`` (``run_turn``'s authoritative control-flow trace); the
recording wrapper additionally proves the backend's store actually received the
finished turn.
"""
from __future__ import annotations

from eval.memory_client import StubMemoryClient

from chat.engine import RECALL_ONLY_NOTICE, run_turn


class RecordingBackend:
    """Wraps a backend to capture what the POST-hook stored."""

    def __init__(self, inner):
        self._inner = inner
        self.stored_content: str | None = None
        self.recall_calls = 0

    def recall_facts(self, persona, query, k=24):
        self.recall_calls += 1
        return self._inner.recall_facts(persona, query, k=k)

    def store_facts(self, persona, content, mode="auto"):
        self.stored_content = content
        return self._inner.store_facts(persona, content, mode=mode)


def _seed(persona: str, fact: str) -> StubMemoryClient:
    stub = StubMemoryClient()
    stub.store_facts_verbatim(persona, [fact])
    return stub


def test_pre_recall_inject_reply_post_store_in_order():
    persona = "researcher"
    fact = "The user is allergic to peanuts."
    # Query shares tokens with the fact so the stub's token-overlap recall returns it.
    query = "what is the user allergic to?"
    backend = RecordingBackend(_seed(persona, fact))
    seen_prompts: list[str] = []

    def fake_llm(prompt: str, system_prompt: str) -> str:
        seen_prompts.append(prompt)
        assert system_prompt  # the reply carries an instruction system prompt
        return "You are allergic to peanuts."

    result = run_turn(backend, persona, query, llm_fn=fake_llm)

    # PRE-hook, then the reply, then the POST-hook write — in that order.
    assert result.events == ["recall", "llm", "store"]
    assert backend.recall_calls == 1

    # PRE-hook recalled the seeded fact and injected it into the LLM prompt.
    assert any(f.summary == fact for f in result.recalled)
    assert fact in result.context_block
    assert fact in seen_prompts[0]
    assert f"User: {query}" in seen_prompts[0]

    # Reply is what the LLM produced.
    assert result.reply == "You are allergic to peanuts."
    assert result.llm_used is True

    # POST-hook stored the finished turn (both sides of it).
    assert backend.stored_content is not None
    assert f"User: {query}" in backend.stored_content
    assert "Agent: You are allergic to peanuts." in backend.stored_content


def test_degrades_to_recall_only_when_no_llm():
    persona = "researcher"
    backend = RecordingBackend(_seed(persona, "The user drives a 2019 Leaf."))

    # llm_fn returns None === memory.llm.llm_call with no credential/SDK available.
    result = run_turn(backend, persona, "what car does the user drive?", llm_fn=lambda p, s: None)

    # Full loop still ran; the reply degraded to the recall-only notice.
    assert result.events == ["recall", "llm", "store"]
    assert result.llm_used is False
    assert result.reply == RECALL_ONLY_NOTICE
    # POST-hook still fired (production Stop always writes; extraction may no-op).
    assert backend.stored_content is not None


def test_empty_recall_yields_placeholder_block():
    backend = StubMemoryClient()  # nothing seeded
    result = run_turn(backend, "researcher", "anything?", llm_fn=lambda p, s: "ok")
    assert result.recalled == []
    assert "no relevant memories" in result.context_block
    assert result.events == ["recall", "llm", "store"]


# ── recall superset: created_at (WRITE/RECENCY time) is distinct from occurred_at ──

class _RowBackend:
    """Backend returning a hand-crafted recall row (the EngineBackend superset)."""

    def __init__(self, rows):
        self._rows = rows

    def recall_facts(self, persona, query, k=24):
        return self._rows

    def store_facts(self, persona, content, mode="auto"):
        return {"written": 0, "mode": mode}


def test_recall_carries_event_and_write_times_distinctly():
    backend = _RowBackend([{
        "summary": "a fact", "scope": "own", "score": 1103.0, "matched": True,
        "hit_count": 4, "occurred_at": "2026-07-10", "created_at": "2026-07-20",
    }])
    result = run_turn(backend, "researcher", "q", llm_fn=lambda p, s: "ok")
    f = result.recalled[0]
    assert f.occurred_at == "2026-07-10"  # EVENT time
    assert f.created_at == "2026-07-20"   # WRITE / RECENCY time — surfaced separately
    assert f.scope == "own" and f.matched is True and f.hit_count == 4


# ── model dropdown: the chosen model reaches BOTH the reply and the store ───────

class _ModelBackend:
    """store_facts accepts a model kwarg and records what it received."""

    def __init__(self):
        self.store_model = "UNSET"

    def recall_facts(self, persona, query, k=24):
        return []

    def store_facts(self, persona, content, mode="auto", model=None):
        self.store_model = model
        return {"written": 0, "mode": mode}


class _LeanBackend:
    """store_facts has NO model param (the stub contract) — proves run_turn omits
    the kwarg when no model is selected."""

    def recall_facts(self, persona, query, k=24):
        return []

    def store_facts(self, persona, content, mode="auto"):
        return {"written": 0, "mode": mode}


def test_selected_model_threads_to_store():
    backend = _ModelBackend()
    run_turn(backend, "researcher", "hi", model="opus", llm_fn=lambda p, s: "x")
    assert backend.store_model == "opus"


def test_no_model_omits_kwarg_for_lean_backends():
    # A backend whose store_facts takes no model must still work when model is None.
    run_turn(_LeanBackend(), "researcher", "hi", model=None, llm_fn=lambda p, s: "x")


def test_selected_model_threads_to_reply(monkeypatch):
    recorded: dict = {}

    def fake_llm_call(prompt, *, system_prompt="", model=None, max_tokens=256):
        recorded["model"] = model
        return "a reply"

    monkeypatch.setattr("memory.llm.llm_call", fake_llm_call)
    backend = _ModelBackend()
    # llm_fn=None -> the default reply fn, which must pass the chosen model through.
    result = run_turn(backend, "researcher", "hi", model="sonnet")
    assert recorded["model"] == "sonnet"      # reply used the selected model
    assert backend.store_model == "sonnet"    # and so did the post-hook store
    assert result.reply == "a reply" and result.llm_used is True
