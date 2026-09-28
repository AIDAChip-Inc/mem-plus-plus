"""Headless tests for the System / Architecture tab.

The tab is presentation-only (no engine import, no DB, no API key), so these run
anywhere. They prove the self-contained HTML/CSS diagram renders every key node /
constant from ARCHITECTURE.md, that it pulls in NO CDN / mermaid asset, and that
build_demo() wires the tab in.
"""
from __future__ import annotations

from chat.system_tab import (
    SYS_CSS,
    _LIFECYCLE_MD,
    _READING_MD,
    _SUMMARY_MD,
    render_system_diagram,
)


def test_diagram_contains_hook_loop_nodes():
    out = render_system_diagram()
    for label in ("PRE-hook", "REPLY", "POST-hook", "recall_facts", "store_facts",
                  "run_turn"):
        assert label in out, f"missing hook-loop node: {label}"


def test_diagram_contains_recall_internals():
    out = render_system_diagram()
    for label in ("Scope filter FIRST", "structured / recent", "lexical / fuzzy",
                  "entity-tag", "vector-ANN", "MiniLM-384", "Weighted RRF", "K=60",
                  "1 / 1 / 4", "MATCH → HITS → RECENCY", "Recent-reserve",
                  "top-k = 24", "hit-writeback", "OWN", "TEAM"):
        assert label in out, f"missing recall-pipeline node: {label}"


def test_diagram_contains_write_pipeline():
    out = render_system_diagram()
    for label in ("Extraction", "store_facts_verbatim", "content_tsv", "Insert row",
                  "agent_summary EXCLUDED", "valid_from"):
        assert label in out, f"missing write-pipeline node: {label}"


def test_diagram_contains_consolidation_and_store():
    out = render_system_diagram()
    for label in ("consolidate", "τ = 0.85", "dedup_groups", "CONFLICT", "RESTATEMENT",
                  "DISTINCT", "supersede", "is_active = TRUE", "merge",
                  "Postgres 16 + pgvector", "agent_memory", "memory_consolidation_run"):
        assert label in out, f"missing consolidation/store node: {label}"


def test_diagram_contains_consumers():
    out = render_system_diagram()
    for label in ("Chat tab", "Eval harness", "Memory tab"):
        assert label in out, f"missing consumer node: {label}"


def test_strictly_self_contained_no_cdn():
    """No mermaid, no external network fetch anywhere in the tab's HTML/CSS/prose —
    the NO-CDN rule. (Doc-relative links inside the markdown prose are fine.)"""
    blob = render_system_diagram() + SYS_CSS + _READING_MD + _SUMMARY_MD + _LIFECYCLE_MD
    low = blob.lower()
    assert "mermaid" not in low
    assert "http://" not in low and "https://" not in low
    assert "<script" not in low and "cdn" not in low


def test_uses_shared_design_system_classes():
    """The diagram reuses the shared .mr-* system so it reads as one product."""
    out = render_system_diagram()
    assert "mr-panel" in out and "mr-sec-title" in out
    assert "mr-anode" in out and "mr-agroup" in out
    # Horizontal-scroll wrap so the wide diagram never scrolls the page body.
    assert "mr-archwrap" in out
    assert "overflow-x:auto" in SYS_CSS


def test_build_demo_wires_system_tab():
    """build_demo() constructs with the System tab present (no launch)."""
    from chat.app import build_demo

    demo = build_demo()
    assert demo is not None
