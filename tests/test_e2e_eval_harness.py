"""Always-runnable E2E: the eval harness over a tiny real-format LoCoMo slice.

Uses a hand-built ``tiny_locomo.json`` (known gold answers) loaded through the
REAL ``eval.adapters.locomo`` adapter, then runs the full ingest→recall→score
loop with the dependency-free StubMemoryClient (deterministic, no engine, no API
key). Proves the adapter, harness scoring, and results-JSON emission end-to-end.

The stub backend is explicitly a DRY-RUN wiring check (``backend="stub"``), NOT a
reportable score — the reportable ``backend="engine"`` path is DB-gated and, as
of this suite, blocked by a harness↔engine seam defect (see
``test_contract_gaps.py`` and E2E_STATUS.md).
"""
from __future__ import annotations

import json

import pytest

from eval._runner import _stub_answer_fn, _stub_judge_fn
from eval.adapters import locomo
from eval.harness import run_benchmark
from eval.memory_client import StubMemoryClient

from .conftest import TINY_LOCOMO


@pytest.fixture()
def conversations():
    return locomo.load(TINY_LOCOMO)


def test_locomo_adapter_parses_tiny_fixture(conversations):
    """The real adapter maps the fixture faithfully: turn provenance ids, gold
    evidence, and adversarial→abstention all survive."""
    assert len(conversations) == 2
    conv0 = conversations[0]
    assert len(conv0.turns) == 6  # 3 + 3 across two sessions
    assert {t.turn_id for t in conv0.turns} >= {"D1:1", "D2:1", "D2:3"}
    adv = [q for q in conv0.qa if q.abstention]
    assert len(adv) == 1 and not adv[0].gold_turn_ids  # unanswerable, no gold evidence
    single = next(q for q in conv0.qa if q.qa_id == "conv0:q0")
    assert single.gold_turn_ids == {"D1:1"} and single.gold_answers == ["Kyoto"]


def test_harness_recall_only_scores_retrieval_and_emits_json(conversations, tmp_path):
    """recall_only sweep ($0, no answer/judge): the stub retrieves gold turns, so
    recall@k is perfect, adversarial items are excluded from retrieval aggregates,
    and the results JSON round-trips with the expected shape."""
    results = run_benchmark(
        conversations, StubMemoryClient(), recall_only=True, k=24, benchmark="locomo-tiny",
        backend="stub",
    )
    s, cfg = results["summary"], results["config"]
    assert cfg["recall_only"] is True and cfg["backend"] == "stub"
    assert cfg["n_conversations"] == 2 and cfg["n_questions"] == 7
    assert s["recall_at_k"] == 1.0  # stub finds every gold turn
    assert s["token_f1"] is None  # recall-only: no answer scoring
    # Adversarial (no gold evidence) is excluded from the retrieval mean.
    assert results["results"][4]["recall_at_k"] is None

    out = tmp_path / "locomo_tiny_recall.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    reloaded = json.loads(out.read_text())
    assert reloaded["summary"]["recall_at_k"] == 1.0
    assert len(reloaded["results"]) == 7


def test_harness_full_stub_pipeline_runs_answer_and_judge(conversations):
    """Full ingest→recall→answer→judge→score with the deterministic stub answer
    and judge callables — proves the answer/judge plumbing without an API key."""
    results = run_benchmark(
        conversations, StubMemoryClient(),
        answer_fn=_stub_answer_fn(), judge_fn=_stub_judge_fn(),
        k=24, benchmark="locomo-tiny", backend="stub",
    )
    s = results["summary"]
    assert s["token_f1"] is not None  # answer scoring ran
    assert s["judge_accuracy"] is not None  # judge verdicts parsed
    assert s["total_input_tokens"] > 0  # token accounting flowed through
    # Every QA produced a fully-populated row.
    assert all(r["latency_s"] >= 0.0 for r in results["results"])


def test_engine_store_verbatim_accepts_3tuples_on_sqlite(sqlite_public_api):
    """The engine's ACTUAL (de-facto) store_facts_verbatim contract — a list of
    (summary, tags, occurred_at-datetime) 3-tuples — works end-to-end on the
    degraded path. (The harness passes plain strings instead; that seam gap is
    pinned in test_contract_gaps.py.)"""
    from memory import recall

    res = recall.store_facts_verbatim(
        "researcher:locomo_0",
        [("[2023] Alice: I booked my trip to Kyoto", ["kyoto", "trip"], None)],
    )
    assert res["written"] == 1 and res["mode"] == "facts"
    rows = recall.recall_facts("researcher:locomo_0", "Kyoto trip", k=5)
    assert any("Kyoto" in r["summary"] for r in rows)
