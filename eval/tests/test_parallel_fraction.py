"""Fraction sub-sampling + parallelism-invariance tests (no engine, no API key).

Covers the two capabilities added for quick-iteration + speed:
  * ``select_qa_fraction`` — deterministic, reproducible QA sub-sampling.
  * ``workers`` — a ``workers=4`` run yields the SAME aggregate + rows as
    ``workers=1`` on the stub backend (parallelism changes speed only).
"""
from __future__ import annotations

import copy

import pytest

from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import MAX_WORKERS, run_benchmark, select_qa_fraction
from eval.llm import LLMReply
from eval.memory_client import StubMemoryClient


def _conversations(n_conv: int = 5, n_qa: int = 4) -> list[Conversation]:
    """A small multi-conversation fixture: each conversation has distinct facts
    so recall is deterministic under the lexical stub."""
    convs: list[Conversation] = []
    colors = ["blue", "red", "green", "amber", "violet", "cyan", "teal", "coral"]
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


# --- fraction selection (deterministic sub-sampling) ------------------------

def test_fraction_one_is_passthrough():
    convs = _conversations()
    out = select_qa_fraction(convs, 1.0)
    assert [c.conversation_id for c in out] == [c.conversation_id for c in convs]
    assert sum(len(c.qa) for c in out) == sum(len(c.qa) for c in convs)


def test_fraction_selects_ceil_of_total_qa():
    convs = _conversations(n_conv=5, n_qa=4)  # 20 QA items
    out = select_qa_fraction(convs, 0.25)
    # ceil(0.25 * 20) == 5 items kept, across whichever conversations own them.
    assert sum(len(c.qa) for c in out) == 5


def test_fraction_is_deterministic():
    convs = _conversations()
    a = select_qa_fraction(convs, 0.3)
    b = select_qa_fraction(convs, 0.3)
    ids_a = [(c.conversation_id, q.qa_id) for c in a for q in c.qa]
    ids_b = [(c.conversation_id, q.qa_id) for c in b for q in c.qa]
    assert ids_a == ids_b


def test_fraction_keeps_natural_order_and_full_turns():
    convs = _conversations()
    out = select_qa_fraction(convs, 0.5)
    for c in out:
        # turns are never sub-sampled (ingest needs the whole conversation)
        assert len(c.turns) == 3
        qa_ids = [q.qa_id for q in c.qa]
        assert qa_ids == sorted(qa_ids)  # natural (stable) order preserved


def test_tiny_fraction_keeps_at_least_one():
    convs = _conversations()
    out = select_qa_fraction(convs, 0.001)
    assert sum(len(c.qa) for c in out) == 1


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_fraction_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        select_qa_fraction(_conversations(), bad)


def test_run_records_fraction_metadata():
    convs = _conversations(n_conv=4, n_qa=5)  # 20 QA
    res = run_benchmark(convs, StubMemoryClient(), recall_only=True,
                        benchmark="t", sample_fraction=0.5)
    assert res["config"]["sample_fraction"] == 0.5
    assert res["config"]["sample_seed"] is not None
    assert res["config"]["n_questions"] == 10  # ceil(0.5 * 20)


# --- parallelism invariance -------------------------------------------------

_DETERMINISTIC_SUMMARY_KEYS = [
    "recall_at_k", "mrr", "ndcg_at_k", "token_f1", "exact_match",
    "judge_accuracy", "total_input_tokens", "total_output_tokens",
    "est_cost_usd", "storage_bytes", "per_category",
]


def _strip_latency(res: dict) -> dict:
    """Drop wall-clock fields that legitimately vary with load, so we compare
    only the deterministic metrics that parallelism must NOT change."""
    out = copy.deepcopy(res)
    # Latency is wall-clock and legitimately varies with load; the split
    # search/total percentiles added for the mem0-style tables are the same
    # kind of value, so they are excluded from the invariance comparison too.
    # `workers` is echoed into the summary as systems provenance (a latency is
    # only interpretable next to the parallelism that produced it). It is a
    # description of the RUN, not a metric, and these two runs differ in it by
    # construction -- so it is dropped alongside the wall-clock fields.
    for _k in ("mean_latency_s", "mean_total_latency_s",
               "latency_search", "latency_total", "latency_answer", "workers"):
        out["summary"].pop(_k, None)
    for row in out["results"]:
        row.pop("latency_s", None)
        row.pop("total_latency_s", None)
        row.pop("answer_latency_s", None)
    return out


def test_workers_invariance_full_run():
    convs = _conversations(n_conv=5, n_qa=4)
    serial = run_benchmark(
        _conversations(n_conv=5, n_qa=4), StubMemoryClient(),
        answer_fn=_echo_answer_fn, judge_fn=_lenient_judge_fn, benchmark="t", workers=1,
    )
    parallel = run_benchmark(
        _conversations(n_conv=5, n_qa=4), StubMemoryClient(),
        answer_fn=_echo_answer_fn, judge_fn=_lenient_judge_fn, benchmark="t", workers=4,
    )
    assert _strip_latency(serial)["summary"] == _strip_latency(parallel)["summary"]
    # Row order + content identical (map preserves input order).
    assert _strip_latency(serial)["results"] == _strip_latency(parallel)["results"]
    assert serial["config"]["workers"] == 1
    assert parallel["config"]["workers"] == 4
    assert len(convs) == 5  # fixture sanity


def test_workers_invariance_with_fraction():
    kw = dict(answer_fn=_echo_answer_fn, judge_fn=_lenient_judge_fn,
              benchmark="t", sample_fraction=0.5)
    serial = run_benchmark(_conversations(n_conv=6, n_qa=4), StubMemoryClient(),
                           workers=1, **kw)
    parallel = run_benchmark(_conversations(n_conv=6, n_qa=4), StubMemoryClient(),
                             workers=4, **kw)
    # Aggregate + rows must match; config.workers legitimately differs (1 vs 4).
    s, p = _strip_latency(serial), _strip_latency(parallel)
    assert s["summary"] == p["summary"]
    assert s["results"] == p["results"]


def test_workers_clamped_to_max(capsys):
    res = run_benchmark(_conversations(n_conv=3), StubMemoryClient(), recall_only=True,
                        benchmark="t", workers=MAX_WORKERS + 100)
    assert res["config"]["workers"] == MAX_WORKERS
    assert "clamping" in capsys.readouterr().out


def test_workers_below_one_raises():
    with pytest.raises(ValueError):
        run_benchmark(_conversations(), StubMemoryClient(), recall_only=True,
                      benchmark="t", workers=0)
