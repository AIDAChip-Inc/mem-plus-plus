"""Headless tests for the Memory tab.

The filter → ``browse_memories`` wiring is proved two ways: with a capturing stub
(asserting the exact kwargs the UI controls map to — tags box → list, numbers →
clamped ints, blank → None), and end-to-end against the REAL read-only
``browse_memories`` on the always-available SQLite-degraded path (no Postgres, no
API key). Rendering is checked for escaping + the expected columns.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memory import recall
from memory.models import Base

from chat.memory_tab import (
    DEFAULT_LIMIT,
    _default_persona,
    _persona_choices,
    do_consolidate,
    do_merge,
    do_reset,
    do_supersede,
    on_browse,
    on_merge_default,
    on_supersede_default,
    render_memory_table,
    run_browse,
)


class _CapturingBrowse:
    """Records the kwargs the tab hands to browse_memories, returns nothing."""

    def __init__(self):
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return []


# ── filter → browse_memories wiring (stub) ──────────────────────────────────────

def test_run_browse_maps_all_controls():
    cap = _CapturingBrowse()
    run_browse(
        text="  phase noise  ", semantic=True, persona="  awsi  ", scope="own",
        tags="PLL, pgvector  engine", occurred_after="2026-01-01",
        occurred_before="2026-12-31", min_hit_count=2.0, order="hits",
        limit=10.0, offset=5.0, browse_fn=cap,
    )
    assert cap.kwargs == {
        "text": "phase noise",            # trimmed
        "semantic": True,
        "persona": "awsi",                # trimmed
        "scope": "own",
        "tags": ["PLL", "pgvector", "engine"],  # comma/space split, any-match
        "occurred_after": "2026-01-01",
        "occurred_before": "2026-12-31",
        "min_hit_count": 2,               # float -> clamped int
        "order": "hits",
        "limit": 10,
        "offset": 5,
    }


def test_run_browse_blank_fields_become_none_or_defaults():
    cap = _CapturingBrowse()
    run_browse(text="   ", persona="", tags="", occurred_after="", occurred_before="",
               min_hit_count=None, limit=None, offset=None, browse_fn=cap)
    k = cap.kwargs
    assert k["text"] is None and k["persona"] is None and k["tags"] is None
    assert k["occurred_after"] is None and k["occurred_before"] is None
    assert k["min_hit_count"] == 0        # None -> 0
    assert k["limit"] == DEFAULT_LIMIT    # None -> default
    assert k["offset"] == 0
    assert k["scope"] == "all" and k["order"] == "recency" and k["semantic"] is False


# ── rendering ────────────────────────────────────────────────────────────────

def test_render_table_columns_and_escaping():
    rows = [{
        "content": "PLL <script> filter", "agent_type": "awsi", "scope": "own",
        "tags": ["pll", "phase-noise"], "by": "alice",
        "occurred_at": "2026-07-20T00:00:00+00:00", "hit_count": 3,
    }]
    out = render_memory_table(rows, offset=0)
    assert "&lt;script&gt;" in out and "<script>" not in out  # escaped, not injected
    assert "awsi" in out and "alice" in out and "2026-07-20" in out  # persona/by/event
    assert "pll" in out and "phase-noise" in out                    # tag chips
    assert ">3<" in out                                             # hit_count cell


def test_render_empty_rows_shows_hint():
    out = render_memory_table([], offset=0)
    assert "No memories match" in out


def _full_cell_row() -> dict:
    """A row carrying EVERY key browse._full_cell returns — the full production cell."""
    return {
        "id": "11111111-2222-3333-4444-555555555555",
        "summary": "sched fact", "content": "the scheduler runs nightly",
        "context_summary": "sched fact ctx", "agent_summary": "team-facing sched fact",
        "tags": ["sched"], "by": "alice",
        "user_id": "aaaaaaaa-0000-0000-0000-000000000000", "agent_type": "awsi",
        "project_id": "bbbbbbbb-0000-0000-0000-000000000000",
        "customer_id": "cccccccc-0000-0000-0000-000000000000",
        "session_id": "dddddddd-0000-0000-0000-000000000000", "scope": "own",
        "memory_tier": "working", "discipline": "backend", "authority": "high",
        "occurred_at": "2026-07-10T00:00:00+00:00",
        "created_at": "2026-07-20T09:30:00+00:00",
        "last_used_at": "2026-07-22T11:00:00+00:00",
        "valid_from": "2026-07-10T00:00:00+00:00", "valid_to": None,
        "is_active": True,
        "superseded_by_id": None, "parent_id": None,
        "hit_count": 2, "last_hit_by": "dina", "salience": 1103.5,
        "source_message_id": None,
        "has_embedding": True, "embedding_dim": 384, "has_tsv": True,
    }


def test_render_full_cell_all_columns():
    """The full-cell table renders EVERY agent_memory field as a tooltip'd column —
    grouped, horizontally scrollable, with the cryptic columns explained in title=."""
    from chat.memory_tab import _COLUMN_GROUPS

    out = render_memory_table([_full_cell_row()], offset=0)
    # Every column key from the spec appears as a field-name header.
    keys = [key for _g, cols in _COLUMN_GROUPS for key, _k, _t in cols]
    assert len(keys) == 30  # the full agent_memory cell (summary is derived, not shown)
    for key in keys:
        assert f">{key}</th>" in out, f"missing column header: {key}"
    # Group headers + horizontal-scroll wrap + pinned field-name row are present
    # (the '&' groups render HTML-escaped, so match on unambiguous fragments).
    for group in ("Identity", "Scope coordinates", "Classification", "Temporal",
                  "Lifecycle", "Ranking", "Provenance"):
        assert group in out
    assert 'class="mr-tablewrap"' in out and "mr-fullcell" in out
    assert "mr-grouprow" in out and "mr-colrow" in out
    # Values render: id truncated (prefix + full in title), bi-temporal, is_active,
    # salience, embedding presence/dim.
    assert "11111111" in out and "11111111-2222-3333-4444-555555555555" in out
    assert "2026-07-10 00:00" in out and "2026-07-20 09:30" in out  # event vs written
    assert "true" in out and "1103.5" in out and ">384<" in out
    # The is_active column tooltip explains the recallability contrast.
    assert "Recallable?" in out and "manual retire/merge sets FALSE" in out


def test_render_distinguishes_event_written_last_used():
    """The full cell keeps the three distinct timestamps as their raw column names —
    occurred_at (event), created_at (written/recency), last_used_at — each shown to
    the minute with the full ISO value preserved in a title."""
    out = render_memory_table([_full_cell_row()], offset=0)
    assert ">occurred_at</th>" in out and ">created_at</th>" in out
    assert ">last_used_at</th>" in out
    assert "2026-07-10 00:00" in out and "2026-07-20 09:30" in out
    assert "2026-07-22 11:00" in out
    assert 'title="2026-07-10T00:00:00+00:00"' in out  # full ISO preserved in title


# ── action menu (supersede / merge / reset) — stub the mutate ops ───────────────

def _boom(*a, **k):
    raise AssertionError("mutate op must NOT be called on a failed validation")


def test_do_supersede_calls_fn_per_selected_with_reason():
    calls = []

    def fake(mid, *, superseded_by=None, reason=None):
        calls.append((mid, reason))
        return {"id": mid, "superseded_by": None, "valid_to": "2026-07-23T00:00:00+00:00"}

    ok, out = do_supersede(["m1", "m2", "m1"], "duplicate", True, supersede_fn=fake)
    assert ok is True
    assert calls == [("m1", "duplicate"), ("m2", "duplicate")]  # de-duped, reason threaded
    assert "m1" in out and "m2" in out and "Superseded" in out


def test_do_supersede_requires_selection_and_confirm():
    ok, out = do_supersede([], "", True, supersede_fn=_boom)
    assert ok is False and "Nothing selected" in out
    ok, out = do_supersede(["m1"], "", False, supersede_fn=_boom)
    assert ok is False and "Confirm" in out


def test_do_merge_calls_with_ids_and_summary():
    seen = {}

    def fake(ids, *, summary=None):
        seen["ids"], seen["summary"] = ids, summary
        return {"parent_id": "PARENT", "merged_count": len(ids), "child_ids": ids}

    ok, out = do_merge(["a", "b", "c"], "  union text  ", True, merge_fn=fake)
    assert ok is True
    assert seen["ids"] == ["a", "b", "c"] and seen["summary"] == "union text"
    assert "PARENT" in out and "Merged" in out


def test_do_merge_requires_two_and_confirm():
    ok, out = do_merge(["only-one"], "", True, merge_fn=_boom)
    assert ok is False and "at least two" in out.lower()
    ok, out = do_merge(["a", "b"], "", False, merge_fn=_boom)
    assert ok is False and "Confirm" in out


def test_do_merge_blank_summary_passes_none():
    seen = {}

    def fake(ids, *, summary=None):
        seen["summary"] = summary
        return {"parent_id": "P", "merged_count": 2, "child_ids": ids}

    do_merge(["a", "b"], "   ", True, merge_fn=fake)
    assert seen["summary"] is None  # blank -> deterministic union in the engine


def test_do_reset_requires_confirm_then_reports_counts():
    ok, out = do_reset(False, reset_fn=_boom)
    assert ok is False and "Confirm" in out
    ok, out = do_reset(True, reset_fn=lambda: {"deleted_memories": 5, "deleted_tags": 3})
    assert ok is True and ">5<" in out and ">3<" in out


def test_do_consolidate_calls_fn_with_persona_and_threshold():
    seen = {}

    def fake(persona, *, threshold=0.85, max_groups=0):
        seen["persona"], seen["threshold"] = persona, threshold
        return {"run_id": "RUN-1", "pools_merged": 2, "memories_before": 9,
                "memories_after": 6, "superseded": 1}

    ok, out = do_consolidate("  researcher  ", 0.9, True, consolidate_fn=fake)
    assert ok is True
    assert seen == {"persona": "researcher", "threshold": 0.9}  # trimmed, threshold threaded
    # The full contract dict is rendered in the result panel.
    assert "Consolidated" in out and "RUN-1" in out
    assert ">2<" in out and ">9<" in out and ">6<" in out and ">1<" in out


def test_do_consolidate_default_threshold_is_085():
    seen = {}

    def fake(persona, *, threshold=0.85, max_groups=0):
        seen["threshold"] = threshold
        return {"run_id": "R", "pools_merged": 0, "memories_before": 0,
                "memories_after": 0, "superseded": 0}

    do_consolidate("researcher", None, True, consolidate_fn=fake)  # blank threshold -> default
    assert seen["threshold"] == 0.85


def test_do_consolidate_requires_persona_and_confirm():
    ok, out = do_consolidate("", 0.85, True, consolidate_fn=_boom)
    assert ok is False and "Persona required" in out
    ok, out = do_consolidate("researcher", 0.85, False, consolidate_fn=_boom)
    assert ok is False and "Confirm" in out


# ── Consolidate persona dropdown — derived from the personas PRESENT in the DB ───

def test_persona_choices_distinct_sorted_from_browse():
    """The Consolidate dropdown derives its options from a read-only scope=all browse,
    collecting the DISTINCT agent_type values (blanks dropped, sorted)."""
    seen = {}

    def stub(**kwargs):
        seen.update(kwargs)
        return [{"agent_type": "dina"}, {"agent_type": "awsi"},
                {"agent_type": "dina"}, {"agent_type": None}, {}]

    assert _persona_choices(browse_fn=stub) == ["awsi", "dina"]  # distinct + sorted
    assert seen["scope"] == "all"  # whole project pool, not persona-scoped


def test_persona_choices_empty_on_db_error():
    """A DB-down browse degrades to an empty list — the dropdown stays typeable
    (allow_custom_value) rather than crashing the tab build."""
    def boom(**kwargs):
        raise RuntimeError("db down / not migrated")

    assert _persona_choices(browse_fn=boom) == []


def test_default_persona_prefers_researcher_else_first():
    assert _default_persona(["awsi", "researcher", "dina"]) == "researcher"  # present
    assert _default_persona(["awsi", "dina"]) == "awsi"                      # first
    assert _default_persona([]) == "researcher"                             # custom fallback


def test_do_action_surfaces_error_not_crash():
    def raises(*a, **k):
        raise ValueError("memory X is out of scope")

    ok, out = do_supersede(["x"], "", True, supersede_fn=raises)
    assert ok is False and "out of scope" in out and "Action failed" in out


# ── action menu end-to-end against the REAL mutate ops (SQLite-degraded path) ────

def test_supersede_merge_reset_end_to_end(sqlite_api):
    """do_supersede / do_merge / do_reset drive the REAL memory.mutate ops on the
    degraded path — proving the action menu reuses the shipped WRITE API."""
    from datetime import UTC, datetime

    recall.store_facts_verbatim("awsi", [
        ("Fact one about pgvector", ["pgvector"], datetime(2026, 7, 20, tzinfo=UTC)),
        ("Fact two about pgvector", ["pgvector"], datetime(2026, 7, 21, tzinfo=UTC)),
    ])
    ids = [r["id"] for r in run_browse(persona="awsi")]
    assert len(ids) == 2

    # Merge the two into one parent — children archived, parent recallable.
    ok, out = do_merge(ids, "", True)
    assert ok is True and "Merged" in out
    after_merge = run_browse(persona="awsi")
    assert len(after_merge) == 1  # only the active parent remains browsable

    # Supersede the parent — recall no longer returns it.
    parent_id = after_merge[0]["id"]
    ok, out = do_supersede([parent_id], "done", True)
    assert ok is True and "Superseded" in out
    assert run_browse(persona="awsi") == []

    # Reset wipes everything and reports counts.
    recall.store_facts_verbatim("dina", [("something", [], None)])
    ok, out = do_reset(True)
    assert ok is True and "Database reset" in out
    assert run_browse() == []


# ── one-click DEFAULT supersede / merge handlers (SQLite-degraded path) ─────────

def _filters(persona: str) -> tuple:
    """The 11-slot filter tuple the action handlers append after their own args."""
    return ("", False, persona, "all", "", "", "", 0, "recency", DEFAULT_LIMIT, 0)


def test_on_supersede_default_one_click(sqlite_api):
    """The default button retires the selected row(s) in ONE click — no reason, no
    confirm tick (the press is the intent) — via do_supersede(reason="", confirm=True)
    routed through _after_action. Returns the 5-tuple the action targets consume."""
    recall.store_facts_verbatim("awsi", [
        ("Fact A about pgvector", ["pgvector"], datetime(2026, 7, 20, tzinfo=UTC)),
        ("Fact B about pgvector", ["pgvector"], datetime(2026, 7, 21, tzinfo=UTC)),
    ])
    ids = [r["id"] for r in run_browse(persona="awsi")]
    msg, table, sel_upd, rows, persona_upd = on_supersede_default([ids[0]], *_filters("awsi"))
    assert "Superseded" in msg  # the write happened with no confirm tick
    remaining = [r["id"] for r in run_browse(persona="awsi")]
    assert ids[0] not in remaining and len(remaining) == 1  # exactly the picked row retired
    assert "choices" in persona_upd  # Consolidate dropdown refreshed on success


def test_on_merge_default_one_click(sqlite_api):
    """The default button folds the ≥2 selected rows into one parent in ONE click —
    via do_merge(summary=None, confirm=True) (deterministic union) through
    _after_action."""
    recall.store_facts_verbatim("awsi", [
        ("Fact A about pgvector", ["pgvector"], datetime(2026, 7, 20, tzinfo=UTC)),
        ("Fact B about pgvector", ["pgvector"], datetime(2026, 7, 21, tzinfo=UTC)),
    ])
    ids = [r["id"] for r in run_browse(persona="awsi")]
    msg, table, sel_upd, rows, persona_upd = on_merge_default(ids, *_filters("awsi"))
    assert "Merged" in msg
    assert len(run_browse(persona="awsi")) == 1  # only the active parent remains


# ── end-to-end against the REAL read-only browse (SQLite-degraded path) ─────────

@pytest.fixture()
def sqlite_api(monkeypatch):
    """Bind memory.db to a fresh in-memory SQLite so the real write + browse run
    end-to-end on the DEGRADED path (no embeddings, LIKE lexical)."""
    monkeypatch.setenv("MEMORY_USER", "alice")
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    yield engine
    engine.dispose()


def test_run_browse_end_to_end_real_browse(sqlite_api):
    """run_browse -> the real browse_memories -> rows, then render — proving the tab
    reuses the shipped read-only query API, not a reimplementation."""
    recall.store_facts_verbatim("awsi", [
        ("The PLL loop filter reduces phase noise", ["pll"], datetime(2026, 7, 20, tzinfo=UTC)),
        ("Adopt pgvector for semantic recall", ["pgvector"], datetime(2026, 7, 22, tzinfo=UTC)),
    ])
    rows = run_browse(persona="awsi", order="occurred")
    assert [r["summary"] for r in rows] == [
        "Adopt pgvector for semantic recall",             # occurred desc
        "The PLL loop filter reduces phase noise",
    ]
    # A text filter narrows via the real lexical leg; render is well-formed.
    filtered = run_browse(persona="awsi", text="pgvector")
    assert {r["summary"] for r in filtered} == {"Adopt pgvector for semantic recall"}
    assert "Adopt pgvector for semantic recall" in render_memory_table(filtered, offset=0)


def test_on_browse_renders_rows_end_to_end(sqlite_api):
    """on_browse (the Gradio click handler) returns a rendered table, the selection
    dropdown update, and the rows state over the real browse — the full UI event
    path, read-only."""
    recall.store_facts_verbatim("dina", [("Dina designed the schema", ["schema"], None)])
    html_out, choices_upd, rows = on_browse(
        "", False, "dina", "all", "", "", "", 0, "recency", DEFAULT_LIMIT, 0)
    assert "Dina designed the schema" in html_out and "dina" in html_out
    # Selection dropdown is populated with (label, id) options for the action menu.
    assert rows and rows[0]["summary"] == "Dina designed the schema"
    labels = [lbl for lbl, _val in choices_upd["choices"]]
    assert any("Dina designed the schema" in lbl for lbl in labels)
