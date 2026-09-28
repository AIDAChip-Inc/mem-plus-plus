"""Consolidation-quality eval: retention math + sweep loop (stub) + a DB-gated
real-engine sweep that SKIPS cleanly offline.

The pure retention/reduction computation and the sweep orchestration are proven
here with NO engine, NO DB, and NO API key (StubMemoryClient + the stub
consolidate double). The real ONNX+pgvector sweep is a single ``@pytest.mark.db``
test that skips when ``MEMORY_DATABASE_URL`` is unset — matching the tests/ suite's
gating philosophy (never pass vacuously; fail loudly only when the DB is present).
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from eval.consolidation_eval import (
    Corpus,
    ConsolidationQuery,
    build_default_corpus,
    compute_retention,
    make_stub_consolidate,
    reduction_pct,
    recall_for_query,
    run_consolidation_eval,
    _needles_in_top_k,
)
from eval.memory_client import StubMemoryClient


# ── compute_retention (pure) ─────────────────────────────────────────────────────
def test_retention_identical_snapshots_is_one_and_no_loss():
    before = {"a": 1.0, "b": 1.0, "c": 0.5}
    ret = compute_retention(before, dict(before))
    assert ret.retention == 1.0
    assert ret.lost_recall_qids == []
    assert ret.dropped_recall_qids == []
    assert ret.n_scored == 3


def test_retention_total_loss_is_zero_and_flags_every_query():
    before = {"a": 1.0, "b": 1.0}
    after = {"a": 0.0, "b": 0.0}
    ret = compute_retention(before, after)
    assert ret.retention == 0.0
    assert ret.lost_recall_qids == ["a", "b"]


def test_retention_partial_loss_only_flags_zeroed_queries():
    before = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    after = {"a": 1.0, "b": 0.0, "c": 0.5, "d": 1.0}  # b lost, c dropped-but-not-lost
    ret = compute_retention(before, after)
    assert ret.lost_recall_qids == ["b"]            # only the query that hit 0
    assert set(ret.dropped_recall_qids) == {"b", "c"}  # any decrease
    assert ret.retention == pytest.approx((1.0 + 0.0 + 0.5 + 1.0) / 4 / 1.0)


def test_retention_divide_by_zero_guard_when_nothing_retrievable_before():
    # before all 0 -> 0/0 is "nothing to lose", pinned to 1.0 (never a ZeroDivision).
    ret = compute_retention({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0})
    assert ret.retention == 1.0
    assert ret.lost_recall_qids == []


def test_retention_ignores_queries_without_gold_in_either_snapshot():
    before = {"a": 1.0, "b": None}
    after = {"a": 1.0, "b": None}
    ret = compute_retention(before, after)
    assert ret.n_scored == 1  # only 'a' is scored


# ── reduction_pct (pure) ─────────────────────────────────────────────────────────
def test_reduction_pct_basic():
    assert reduction_pct(58, 46) == pytest.approx(100.0 * 12 / 58)


def test_reduction_pct_empty_pool_is_zero():
    assert reduction_pct(0, 0) == 0.0


def test_reduction_pct_no_merge_is_zero():
    assert reduction_pct(58, 58) == 0.0


# ── content-needle recall ─────────────────────────────────────────────────────────
def test_needles_in_top_k_respects_truncation():
    recalled = [{"summary": "irrelevant"}] * 5 + [{"summary": "has NEEDLE-42 inside"}]
    # NEEDLE at rank 6 is invisible at k=5, visible at k=6.
    assert _needles_in_top_k(recalled, ["NEEDLE-42"], k=5) == []
    assert _needles_in_top_k(recalled, ["NEEDLE-42"], k=6) == ["NEEDLE-42"]


def test_recall_for_query_on_stub_hit_and_miss():
    stub = StubMemoryClient()
    stub.store_facts_verbatim("p", ["the clk_main clock runs at 1.833 GHz here"])
    hit = ConsolidationQuery("q", "what clock frequency", ("1.833 GHz",), "cluster")
    miss = ConsolidationQuery("q2", "what clock frequency", ("2.400 GHz",), "cluster")
    assert recall_for_query(stub, "p", hit, k=10) == 1.0
    assert recall_for_query(stub, "p", miss, k=10) == 0.0


# ── make_stub_consolidate double ───────────────────────────────────────────────────
def test_stub_consolidate_merges_near_dups_and_preserves_needle():
    stub = StubMemoryClient()
    stub.store_facts_verbatim("p", [
        "the main clock clk_main runs at 1.833 GHz on the soc",
        "clk_main runs at 1.833 GHz the main clock on the soc",  # near-dup
        "a totally unrelated distractor about dma channels",
    ])
    consolidate = make_stub_consolidate(stub)
    out = consolidate("p", threshold=0.5)
    assert out["memories_before"] == 3
    assert out["pools_merged"] == 1
    assert out["memories_after"] == 2  # one union parent + one singleton
    # union parent still carries the needle -> recall retained
    q = ConsolidationQuery("q", "main clock frequency", ("1.833 GHz",), "cluster")
    assert recall_for_query(stub, "p", q, k=10) == 1.0


# ── build_default_corpus fixture integrity ─────────────────────────────────────────
def test_default_corpus_shape_and_determinism():
    c1 = build_default_corpus(seed=1729)
    c2 = build_default_corpus(seed=1729)
    assert c1.facts == c2.facts  # deterministic under a fixed seed
    assert len(c1.facts) == c1.n_clusters * c1.cluster_size + c1.n_distractors == 58
    d = c1.descriptor
    assert d["n_clusters"] == 6 and d["cluster_size"] == 3 and d["n_distractors"] == 40
    assert d["n_cluster_queries"] == 6 and d["n_distractor_queries"] == 4


def test_default_corpus_cluster_needles_present_distractors_clean():
    c = build_default_corpus()
    cluster_needles = [q.gold_needles[0] for q in c.queries if q.kind == "cluster"]
    corpus_text = "\n".join(c.facts).lower()
    # every cluster needle appears exactly cluster_size times (one per paraphrase)
    for needle in cluster_needles:
        assert corpus_text.count(needle.lower()) == c.cluster_size


# ── sweep loop on the stub (the orchestration) ─────────────────────────────────────
def _small_corpus() -> Corpus:
    """A tiny controlled corpus for a fast stub sweep: 2 clusters (×3, high lexical
    overlap so they group at every swept τ under the stub Jaccard) + 3 distractors."""
    # Each cluster's 3 paraphrases differ by exactly ONE token (Jaccard ~0.78),
    # so they group together at every swept τ <= 0.78 under the stub — the exact
    # merge counts below are then τ-stable and safe to assert precisely.
    facts = [
        "alpha widget frequency is 900 KHZ nominal always",
        "alpha widget frequency is 900 KHZ nominal typically",
        "alpha widget frequency is 900 KHZ nominal sometimes",
        "beta gadget uses part XYZ-77 for control logic block",
        "beta gadget uses part XYZ-77 for control logic module",
        "beta gadget uses part XYZ-77 for control logic unit",
        "gamma unrelated fact one about cooling fans",
        "delta unrelated fact two about power supplies",
        "epsilon unrelated fact three about enclosures",
    ]
    queries = [
        ConsolidationQuery("cluster:alpha", "alpha widget frequency nominal", ("900 KHZ",), "cluster"),
        ConsolidationQuery("cluster:beta", "beta gadget control part", ("XYZ-77",), "cluster"),
        ConsolidationQuery("distractor:gamma", "cooling fans fact", ("cooling fans",), "distractor"),
    ]
    return Corpus(facts=facts, queries=queries, n_clusters=2, cluster_size=3, n_distractors=3)


def test_sweep_loop_stub_reduces_corpus_without_losing_recall():
    stub = StubMemoryClient()
    corpus = _small_corpus()
    results = run_consolidation_eval(
        memory=stub, consolidate_fn=make_stub_consolidate(stub), corpus=corpus,
        taus=(0.5, 0.7), k=10, persona_base="t", backend="stub",
    )
    assert results["config"]["backend"] == "stub"
    assert results["config"]["headline_judge_free"] is True
    assert [r["tau"] for r in results["per_tau"]] == [0.5, 0.7]
    for row in results["per_tau"]:
        # 2 clusters of 3 collapse -> 2 parents, 4 rows removed of 9 = 44.44%
        assert row["pools_merged"] == 2
        assert row["memories_before"] == 9
        assert row["memories_after"] == 5
        assert row["corpus_reduction_pct"] == pytest.approx(100.0 * 4 / 9, abs=0.01)
        # deterministic union preserves every needle -> recall fully retained
        assert row["recall_at_k_before"] == 1.0
        assert row["recall_at_k_after"] == 1.0
        assert row["recall_retention"] == 1.0
        assert row["lost_recall_qids"] == []


def test_sweep_loop_isolates_each_tau_in_its_own_scope():
    # Each τ writes to <base>__t<τ>__r0, so seeding one τ never leaks into another.
    stub = StubMemoryClient()
    run_consolidation_eval(
        memory=stub, consolidate_fn=make_stub_consolidate(stub), corpus=_small_corpus(),
        taus=(0.5, 0.9), k=10, persona_base="iso", backend="stub",
    )
    # Each τ seeded + consolidated its OWN scope independently (a fresh 9-fact
    # corpus each), so both personas exist and reflect their OWN τ outcome: at
    # τ=0.5 both clusters group under the stub Jaccard (→5), at τ=0.9 neither does
    # (paraphrase Jaccard ~0.78 < 0.9 → 9 rows untouched) — proving no cross-leak.
    assert "iso__t0.5__r0" in stub._store
    assert "iso__t0.9__r0" in stub._store
    assert len(stub._store["iso__t0.5__r0"]) == 5  # 2 clusters merged + 3 distractors
    assert len(stub._store["iso__t0.9__r0"]) == 9  # nothing merged at the strict τ


def test_sweep_multirun_reports_stddev_fields_and_is_deterministic():
    stub = StubMemoryClient()
    results = run_consolidation_eval(
        memory=stub, consolidate_fn=make_stub_consolidate(stub), corpus=_small_corpus(),
        taus=(0.5,), k=10, persona_base="mr", n_runs=3, backend="stub",
    )
    row = results["per_tau"][0]
    assert row["n_runs"] == 3
    # judge-free stub path is deterministic -> zero variance across runs
    assert row["recall_retention_stddev"] == 0.0
    assert row["corpus_reduction_stddev"] == 0.0


def test_run_consolidation_eval_rejects_bad_n_runs():
    with pytest.raises(ValueError):
        run_consolidation_eval(memory=StubMemoryClient(),
                               consolidate_fn=make_stub_consolidate(StubMemoryClient()),
                               corpus=_small_corpus(), n_runs=0, backend="stub")


# ── DB-gated: real ONNX + pgvector + real consolidate() sweep ──────────────────────
@pytest.mark.db
@pytest.mark.embedder
def test_real_engine_sweep_is_judge_free_recall_lossless(monkeypatch):
    """The full sweep over the REAL engine: seed with the ONNX embedder, run
    ``memory.consolidation.consolidate`` at each τ, and prove the judge-free
    contract — corpus shrinks (reduction > 0 at the low τ) while recall@k retention
    stays at 1.0 with NO lost-recall cases (the deterministic-union path holds
    without an API key). SKIPS when MEMORY_DATABASE_URL is unset."""
    dsn = os.environ.get("MEMORY_DATABASE_URL")
    if not dsn or dsn.startswith("sqlite"):
        pytest.skip("MEMORY_DATABASE_URL unset/sqlite — real-engine sweep skipped")

    from memory.models import Base

    monkeypatch.setenv("MEMORY_USER", "qiyas-consol-eval")
    engine = create_engine(dsn, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:  # unreachable server
        pytest.skip(f"Postgres unreachable ({exc}) — real-engine sweep skipped")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)

    from eval.memory_client import MemoryClient

    results = run_consolidation_eval(
        memory=MemoryClient(), corpus=build_default_corpus(), taus=(0.80, 0.95),
        k=10, persona_base="pytest_consol", backend="engine",
    )
    rows = {r["tau"]: r for r in results["per_tau"]}
    # low τ merges at least one cluster; retention holds and nothing is lost.
    assert rows[0.80]["pools_merged"] >= 1
    assert rows[0.80]["corpus_reduction_pct"] > 0.0
    for row in results["per_tau"]:
        assert row["recall_retention"] == 1.0, row
        assert row["lost_recall_qids"] == []
    engine.dispose()
