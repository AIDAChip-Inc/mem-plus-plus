"""LongMemEval question_date: adapter -> as-of recall bound -> answer prompt.

Offline: stub memory, stub answerer, stub judge.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from eval.adapters import longmemeval
from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import as_of_bound, run_benchmark
from eval.llm import ANSWER_PROMPT_TEMPLATE, LLMReply, build_answer_prompt
from eval.memory_client import StubMemoryClient

_QDATE = datetime(2023, 5, 30, 23, 40, tzinfo=UTC)


def test_adapter_carries_question_date(tmp_path):
    item = {
        "question_id": "q1", "question_type": "temporal-reasoning",
        "question": "how many days ago?", "answer": "10",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_sessions": [[{"role": "user", "content": "hi", "has_answer": True}]],
        "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "answer_session_ids": ["s1"],
    }
    undated = {**item, "question_id": "q2"}
    del undated["question_date"]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps([item, undated]))
    c1, c2 = longmemeval.load(p)
    assert c1.qa[0].question_date == _QDATE
    assert c2.qa[0].question_date is None


def test_answer_prompt_current_date():
    assert build_answer_prompt(context="ctx", question="Q?") == \
        ANSWER_PROMPT_TEMPLATE.format(context="ctx", question="Q?")
    dated = build_answer_prompt(context="ctx", question="Q?", question_date=_QDATE)
    assert f"Current Date: {_QDATE.isoformat()}\nQuestion: Q?" in dated
    assert dated.count("Current Date:") == 1


class _Recorder(StubMemoryClient):
    def __init__(self):
        super().__init__()
        self.stored: list = []
        self.recall_kwargs: list[dict] = []

    def store_facts_verbatim(self, persona, facts):
        self.stored.extend(facts)
        return super().store_facts_verbatim(persona, facts)

    def recall_facts(self, persona, query, k=24, update_hits=True, **kw):
        self.recall_kwargs.append(kw)
        return super().recall_facts(persona, query, k=k, update_hits=update_hits, **kw)


def _conv(question_date):
    turns = [
        Turn("s1:0", "user", "I adopted a cat named Miso", datetime(2023, 5, 20, tzinfo=UTC)),
        Turn("s2:0", "user", "I adopted a dog named Rex", datetime(2023, 6, 5, tzinfo=UTC)),
        # Same calendar day as the question, later in the day: still eligible.
        Turn("s3:0", "user", "I adopted a fish named Bubbles",
             datetime(2023, 5, 30, 23, 55, tzinfo=UTC)),
    ]
    qa = [QAItem("q", "which pet did I adopt?", ["cat"], "temporal_reasoning",
                 {"s1:0"}, question_date=question_date)]
    return Conversation("c", turns, qa)


def test_harness_bounds_recall_and_dates_prompt():
    prompts: list[str] = []

    def answer(prompt):
        prompts.append(prompt)
        return LLMReply("cat", 1, 1)

    mem = _Recorder()
    res = run_benchmark([_conv(_QDATE)], mem, answer_fn=answer,
                        judge_fn=lambda p: "CORRECT", benchmark="t")
    # Each turn is stored with its session date as occurred_at.
    assert mem.stored == [
        {"summary": t.render(), "occurred_at": t.occurred_at.isoformat()}
        for t in _conv(_QDATE).turns
    ]
    assert mem.recall_kwargs == [{"occurred_before": as_of_bound(_QDATE)}]
    # The turn dated after the question is not recalled.
    assert sorted(res["results"][0]["retrieved_turn_ids"]) == ["s1:0", "s3:0"]
    assert f"Current Date: {_QDATE.isoformat()}" in prompts[0]
    assert "Rex" not in prompts[0]
    assert res["config"]["n_questions_dated"] == 1


def test_harness_undated_unchanged():
    prompts: list[str] = []

    def answer(prompt):
        prompts.append(prompt)
        return LLMReply("cat", 1, 1)

    mem = _Recorder()
    res = run_benchmark([_conv(None)], mem, answer_fn=answer,
                        judge_fn=lambda p: "CORRECT", benchmark="t")
    assert mem.stored == [t.render() for t in _conv(None).turns]  # bare strings
    assert mem.recall_kwargs == [{}]  # no bound keyword sent
    assert "Current Date:" not in prompts[0]
    assert res["config"]["n_questions_dated"] == 0


def test_as_of_bound_is_end_of_question_day():
    assert as_of_bound(_QDATE) == datetime(2023, 5, 30, 23, 59, 59, 999999, tzinfo=UTC)
