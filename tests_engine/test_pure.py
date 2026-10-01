"""Pure-function unit tests — no DB, no network."""
from __future__ import annotations

import importlib
from datetime import UTC, datetime

import pytest

from memory import config
from memory.embeddings.provider import EmbeddingPresetConfig, _is_blank, first_nonblank
from memory.extraction import (
    _guard_unanchored_date,
    _loads_lazy,
    _parse_atomic_response,
    _parse_extraction_response,
    _parse_iso_date,
    normalize_label,
)
from memory.recall import _contributor_slug, _scope
from memory.store import _query_terms, _rrf_fuse, _term_overlap, make_scope_key


# ── RRF (relevance signal) — preserve exact fusion math ────────────────────────
def test_rrf_default_weights_prefer_multi_list_membership():
    fused = _rrf_fuse(([1, 2], [2, 3], []))
    # id 2 appears in both lexical + tag lists -> outscores singletons.
    assert fused[2] > fused[1]
    assert fused[2] > fused[3]


def test_rrf_zero_weight_drops_leg_no_entries():
    fused = _rrf_fuse(([1], [2], [3]), weights=(1.0, 0.0, 1.0))
    assert 2 not in fused  # zeroed leg is a true no-op
    assert set(fused) == {1, 3}


def test_rrf_vector_weight_promotes_vector_members():
    # Same rank in two legs; vector leg weighted 4.0 (production default).
    fused = _rrf_fuse(([1], [], [2]), weights=(1.0, 1.0, 4.0))
    assert fused[2] > fused[1]


def test_rrf_k_controls_decay():
    small_k = _rrf_fuse(([1, 2],), k=1)
    large_k = _rrf_fuse(([1, 2],), k=1000)
    # Smaller k -> sharper top-rank emphasis (bigger gap between ranks 0 and 1).
    assert (small_k[1] - small_k[2]) > (large_k[1] - large_k[2])


def test_rrf_empty():
    assert _rrf_fuse(([], [], [])) == {}


# ── query term extraction / overlap ────────────────────────────────────────────
def test_query_terms_drops_stopwords_and_short():
    assert _query_terms("How did the PLL loop-filter behave") == ["pll", "loop", "filter", "behave"]


def test_query_terms_caps_at_eight():
    assert len(_query_terms(" ".join(f"term{i}" for i in range(20)))) == 8


def test_term_overlap_counts_distinct_terms():
    assert _term_overlap(["pll", "noise"], "the pll had noise", None) == 2
    assert _term_overlap(["pll", "noise"], "unrelated text") == 0


def test_make_scope_key_global_default():
    assert make_scope_key("u", "awsi", None) == "u|awsi|global"
    assert make_scope_key("u", "awsi", "p") == "u|awsi|p"


# ── extraction parsing helpers ─────────────────────────────────────────────────
def test_normalize_label():
    assert normalize_label("  Phase _ Noise ") == "phase-noise"
    assert normalize_label("--A--") == "a"


def test_parse_iso_date_variants():
    assert _parse_iso_date("2023-05-07") == datetime(2023, 5, 7, tzinfo=UTC)
    assert _parse_iso_date("garbage") is None
    assert _parse_iso_date("") is None
    assert _parse_iso_date(None) is None


def test_parse_extraction_response_gated_skip():
    parsed = _parse_extraction_response('{"worth_storing": false, "summary": "", "tags": []}')
    assert parsed == ("", [], None, False, None)


def test_parse_extraction_response_worth_but_empty_is_unusable():
    assert _parse_extraction_response('{"worth_storing": true, "summary": ""}') is None


def test_parse_extraction_response_with_fences_and_agent_summary():
    parsed = _parse_extraction_response(
        '```json\n{"summary": "x", "tags": ["a"], "agent_summary": "agent said y"}\n```'
    )
    assert parsed[0] == "x"
    assert parsed[1] == ["a"]
    assert parsed[4] == "agent said y"


def test_loads_lazy_recovers_array_from_prose():
    assert _loads_lazy('prefix [{"summary": "s"}] suffix') == [{"summary": "s"}]


def test_parse_atomic_wraps_lone_object():
    facts = _parse_atomic_response('{"summary": "one fact", "tags": ["t"]}')
    assert facts == [("one fact", ["t"], None)]


def test_parse_atomic_skips_bad_entries():
    facts = _parse_atomic_response('[{"summary": "ok"}, {"tags": ["x"]}, 42]')
    assert facts == [("ok", [], None)]


def test_guard_unanchored_date_nulls_without_hint():
    d = datetime(2023, 1, 1, tzinfo=UTC)
    # No reference date + no explicit date hint -> model must have invented it.
    assert _guard_unanchored_date(d, "no dates here", None) is None
    # Explicit year hint present -> keep it.
    assert _guard_unanchored_date(d, "back in 2023 we shipped", None) == d
    # Reference date present -> always keep.
    assert _guard_unanchored_date(d, "no dates", datetime(2024, 1, 1, tzinfo=UTC)) == d


# ── embedding provider payload logic ───────────────────────────────────────────
def test_is_blank_and_first_nonblank():
    assert _is_blank(None) and _is_blank("") and _is_blank("   ")
    assert first_nonblank("  ", "content") == "content"  # blank summary doesn't shadow
    assert first_nonblank("summary", "content") == "summary"


def test_build_payload_prefix_and_blank():
    cfg = EmbeddingPresetConfig(model="m", dimensions=384, query_prefix="Q")
    assert cfg.build_payload("hi", "query") == "Q: hi"
    assert cfg.build_payload("hi", "document") == "hi"  # no document_prefix
    assert cfg.build_payload("   ", "query") is None


# ── scope derivation (contract) ────────────────────────────────────────────────
def test_contributor_slug_normalizes(monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "Alice Example")
    assert _contributor_slug() == "alice-example"


def test_scope_is_deterministic_and_uses_memory_user(monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "alice")
    uid, pid, cid, slug = _scope()
    uid2, pid2, cid2, slug2 = _scope()
    assert (uid, pid, cid, slug) == (uid2, pid2, cid2, slug2)
    assert slug == "alice"
    # project_id/customer_id are fixed per the namespace, independent of the user.
    monkeypatch.setenv("MEMORY_USER", "someone-else")
    _, pid3, cid3, _ = _scope()
    assert pid3 == pid and cid3 == cid
    assert _contributor_slug() == "someone-else"


# ── config flags: production defaults + env override ───────────────────────────
def test_atomic_facts_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMORY_ATOMIC_FACTS", raising=False)
    reloaded = importlib.reload(config)
    assert reloaded.MEMORY_ATOMIC_FACTS is False  # matches production registry default
    assert reloaded.MEMORY_EMBEDDINGS_ENABLED is True


def test_flag_env_override(monkeypatch):
    monkeypatch.setenv("MEMORY_ATOMIC_FACTS", "true")
    reloaded = importlib.reload(config)
    assert reloaded.MEMORY_ATOMIC_FACTS is True
    monkeypatch.delenv("MEMORY_ATOMIC_FACTS", raising=False)
    importlib.reload(config)  # restore for other tests


def test_preserved_ranking_constants():
    assert config.MEMORY_RECALL_K == 24
    assert config.MEMORY_RRF_K == 60
    assert (config.MEMORY_RRF_W_FUZZY, config.MEMORY_RRF_W_TAG, config.MEMORY_RRF_W_VECTOR) == (1.0, 1.0, 4.0)
    assert config.RECENCY_HALF_LIFE_DAYS == 7.0
    assert config.MEMORY_RECALL_RECENT_RESERVE == 3
    assert config.CANDIDATE_LIMIT == 50
    assert config.SALIENCE_MATCH_BAND == 1000.0
    assert config.EMBEDDING_DIM == 384


def test_llm_call_soft_fails_without_provider():
    # No anthropic SDK / no credential in the test env -> None, never raises.
    from memory.llm import llm_call

    assert llm_call("prompt") is None
