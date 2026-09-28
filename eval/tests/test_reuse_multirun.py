"""reuse_ingest + run_many + progress_cb tests (no engine, no API key).

Covers the three quick-iteration harness capabilities added on top of
``sample_fraction`` / ``workers``:
  * ``reuse_ingest`` — skips ingest when the scope already holds memories, yet
    yields metrics byte-identical to a fresh ingest of the same turns.
  * ``run_many`` — runs >=2 specs concurrently, order-stable, with a hard
    total-concurrency cap.
  * ``progress_cb`` — fires once per question and reaches done == total (serial
    and parallel).
"""
from __future__ import annotations

import copy
import threading

import pytest

from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import MAX_WORKERS, run_benchmark, run_many
from eval.llm import LLMReply
from eval.memory_client import StubMemoryClient


def _conversations(n_conv: int = 4, n_qa: int = 3) -> list[Conversation]:
    convs: list[Conversation] = []
    colors = ["blue", "red", "green", "amber", "violet", "cyan"]
    for c in range(n_conv):
        color = colors[c % len(colors)]
        turns = [
            Turn(turn_id=f"c{c}t1", speaker="Alice", text=f"my favorite color is {color}"),
            Turn(turn_id=f"c{c}t2", speaker="Bob", text=f"the code word is number {c}"),
            Turn(turn_id=f"c{c}t3", speaker="Alice", text=f"I drive a {color} car"),
        ]
        qa = [
            QAItem(qa_id=f"c{c}q{q}",
                   question=f"what is Alice's favorite color in conversation {c}?",
                   gold_answers=[color], category="single_hop", gold_turn_ids={f"c{c}t1"})
            for q in range(n_qa)
        ]
        convs.append(Conversation(conversation_id=f"conv{c}", turns=turns, qa=qa))
    return convs


def _echo_answer_fn(prompt: str) -> LLMReply:
    for line in prompt.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ": " in line:
            return LLMReply(text=line.split(": ", 1)[1], input_tokens=10, output_tokens=5)
    return LLMReply(text="I don't know", input_tokens=10, output_tokens=2)


def _lenient_judge_fn(prompt: str) -> str:
    return "CORRECT"


def _strip_latency(res: dict) -> dict:
    """Drop wall-clock fields (they vary with load) so we compare only the
    deterministic metrics that reuse_ingest must NOT change."""
    out = copy.deepcopy(res)
    # Latency is wall-clock and legitimately varies with load; the split
    # search/total percentiles added for the mem0-style tables are the same
    # kind of value, so they are excluded from the invariance comparison too.
    for _k in ("mean_latency_s", "mean_total_latency_s",
               "latency_search", "latency_total", "latency_answer"):
        out["summary"].pop(_k, None)
    for row in out["results"]:
        row.pop("latency_s", None)
        row.pop("total_latency_s", None)
        row.pop("answer_latency_s", None)
    return out


# --- reuse_ingest -----------------------------------------------------------

def test_reuse_ingest_skips_ingest_yet_identical_metrics():
    """A reuse run over an already-ingested scope skips the write (proven by the
    stub bucket NOT growing) and produces byte-identical recall + metrics."""
    stub = StubMemoryClient()
    convs = _conversations()

    fresh = run_benchmark(convs, stub, recall_only=True, benchmark="t")
    # 3 turns per conversation were ingested into each conv-scoped persona.
    bucket_sizes = {p: len(f) for p, f in stub._store.items()}
    assert all(size == 3 for size in bucket_sizes.values())
    assert fresh["config"]["reuse_ingest"] is False
    assert fresh["config"]["n_conversations_ingested"] == len(convs)
    assert fresh["config"]["n_conversations_reused"] == 0

    reuse = run_benchmark(convs, stub, recall_only=True, benchmark="t", reuse_ingest=True)
    # Ingest was skipped: no bucket grew (a re-ingest would have doubled them to 6).
    assert {p: len(f) for p, f in stub._store.items()} == bucket_sizes
    assert reuse["config"]["n_conversations_reused"] == len(convs)
    assert reuse["config"]["n_conversations_ingested"] == 0

    # Invariance proof: recall + every aggregate metric byte-identical.
    assert _strip_latency(reuse)["summary"] == _strip_latency(fresh)["summary"]
    assert _strip_latency(reuse)["results"] == _strip_latency(fresh)["results"]


def test_reuse_ingest_full_run_identical():
    """Same invariance with answer+judge enabled (not just recall_only)."""
    stub = StubMemoryClient()
    kw = dict(answer_fn=_echo_answer_fn, judge_fn=_lenient_judge_fn, benchmark="t")
    fresh = run_benchmark(_conversations(), stub, **kw)
    reuse = run_benchmark(_conversations(), stub, reuse_ingest=True, **kw)
    assert _strip_latency(reuse)["summary"] == _strip_latency(fresh)["summary"]
    assert _strip_latency(reuse)["results"] == _strip_latency(fresh)["results"]


def test_reuse_ingest_falls_back_when_empty():
    """reuse_ingest on a FRESH (empty) backend ingests anyway — recall is never
    silently empty — and the config notes the fallback."""
    stub = StubMemoryClient()
    convs = _conversations()
    res = run_benchmark(convs, stub, recall_only=True, benchmark="t", reuse_ingest=True)
    assert res["config"]["reuse_ingest"] is True
    assert res["config"]["n_conversations_ingested"] == len(convs)  # all fell back
    assert res["config"]["n_conversations_reused"] == 0
    assert res["summary"]["recall_at_k"] == 1.0  # recall populated, not empty


# --- run_many ---------------------------------------------------------------

def test_run_many_runs_multiple_specs_order_stable():
    specs = [
        {"conversations": _conversations(n_conv=3), "memory": StubMemoryClient(),
         "recall_only": True, "benchmark": "bench_a", "workers": 2},
        {"conversations": _conversations(n_conv=2), "memory": StubMemoryClient(),
         "recall_only": True, "benchmark": "bench_b", "workers": 2},
    ]
    out = run_many(specs, max_parallel=2)  # peak = 2 + 2 = 4 <= MAX_WORKERS
    assert len(out) == 2
    # Order-stable: result[i] corresponds to spec[i] regardless of finish order.
    assert out[0]["config"]["benchmark"] == "bench_a"
    assert out[1]["config"]["benchmark"] == "bench_b"
    assert out[0]["config"]["n_conversations"] == 3
    assert out[1]["config"]["n_conversations"] == 2


def test_run_many_empty_specs():
    assert run_many([], max_parallel=2) == []


def test_run_many_serial_when_max_parallel_one():
    specs = [
        {"conversations": _conversations(n_conv=2), "memory": StubMemoryClient(),
         "recall_only": True, "benchmark": "bench_a", "workers": MAX_WORKERS},
        {"conversations": _conversations(n_conv=2), "memory": StubMemoryClient(),
         "recall_only": True, "benchmark": "bench_b", "workers": MAX_WORKERS},
    ]
    # active == 1, so peak == MAX_WORKERS (a single run) — allowed, no co-running.
    out = run_many(specs, max_parallel=1)
    assert [r["config"]["benchmark"] for r in out] == ["bench_a", "bench_b"]


def test_run_many_concurrency_cap_raises():
    specs = [
        {"conversations": _conversations(), "memory": StubMemoryClient(),
         "recall_only": True, "benchmark": f"b{i}", "workers": MAX_WORKERS}
        for i in range(2)
    ]
    # Two runs each wanting MAX_WORKERS threads co-running would need 2*MAX_WORKERS.
    with pytest.raises(ValueError, match="concurrency cap"):
        run_many(specs, max_parallel=2)


def test_run_many_bad_max_parallel_raises():
    with pytest.raises(ValueError):
        run_many([{"conversations": _conversations(), "memory": StubMemoryClient(),
                   "recall_only": True}], max_parallel=0)


# --- progress_cb ------------------------------------------------------------

def test_progress_cb_fires_done_equals_total_serial():
    convs = _conversations(n_conv=3, n_qa=4)  # 12 questions
    calls: list[tuple[int, int]] = []
    run_benchmark(convs, StubMemoryClient(), recall_only=True, benchmark="t",
                  progress_cb=lambda done, total: calls.append((done, total)))
    assert len(calls) == 12
    assert all(total == 12 for _, total in calls)
    assert max(done for done, _ in calls) == 12
    # Serial run: monotonically increasing 1..12.
    assert [done for done, _ in calls] == list(range(1, 13))


def test_progress_cb_fires_done_equals_total_parallel():
    convs = _conversations(n_conv=5, n_qa=4)  # 20 questions
    lock = threading.Lock()
    calls: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        with lock:
            calls.append((done, total))

    run_benchmark(convs, StubMemoryClient(), recall_only=True, benchmark="t",
                  workers=4, progress_cb=cb)
    assert len(calls) == 20
    assert all(total == 20 for _, total in calls)
    assert max(done for done, _ in calls) == 20
    # Under the harness lock the emitted `done` values are the full set 1..20.
    assert sorted(done for done, _ in calls) == list(range(1, 21))


def test_progress_cb_none_is_noop():
    # No callback -> no error, results still produced.
    res = run_benchmark(_conversations(), StubMemoryClient(), recall_only=True,
                        benchmark="t", progress_cb=None)
    assert res["config"]["n_questions"] > 0
