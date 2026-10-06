"""End-to-end harness + judge + stub-memory tests (no engine, no API key)."""
from __future__ import annotations

from eval import judge
from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import run_benchmark
from eval.llm import LLMReply
from eval.memory_client import StubMemoryClient


def _conversation() -> Conversation:
    turns = [
        Turn(turn_id="t1", speaker="Alice", text="my favorite color is blue"),
        Turn(turn_id="t2", speaker="Bob", text="the weather is sunny today"),
        Turn(turn_id="t3", speaker="Alice", text="I drive a red car"),
    ]
    qa = [
        QAItem(qa_id="q1", question="what is Alice's favorite color?",
               gold_answers=["blue"], category="single_hop", gold_turn_ids={"t1"}),
        QAItem(qa_id="q2", question="what car does Alice drive?",
               gold_answers=["red car", "a red car"], category="single_hop", gold_turn_ids={"t3"}),
    ]
    return Conversation(conversation_id="c0", turns=turns, qa=qa)


def _echo_answer_fn(prompt: str) -> LLMReply:
    # Echo the top recalled excerpt's text (after the "N. [stamp] speaker: " prefix).
    for line in prompt.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ": " in line:
            return LLMReply(text=line.split(": ", 1)[1], input_tokens=10, output_tokens=5)
    return LLMReply(text="I don't know", input_tokens=10, output_tokens=2)


def _lenient_judge_fn(prompt: str) -> str:
    return "CORRECT"


def test_recall_only_run():
    res = run_benchmark([_conversation()], StubMemoryClient(), recall_only=True, benchmark="t")
    assert res["config"]["n_questions"] == 2
    # The stub lexically ranks the blue/red turns to the top for their queries.
    assert res["summary"]["recall_at_k"] == 1.0
    assert res["summary"]["mrr"] == 1.0
    assert res["summary"]["token_f1"] is None  # answer skipped
    assert res["summary"]["storage_bytes"] > 0


def test_full_run_with_stubs():
    res = run_benchmark(
        [_conversation()], StubMemoryClient(),
        answer_fn=_echo_answer_fn, judge_fn=_lenient_judge_fn, benchmark="t",
    )
    s = res["summary"]
    assert s["judge_accuracy"] == 1.0
    assert s["token_f1"] is not None and s["token_f1"] > 0
    assert s["total_output_tokens"] == 10  # 2 questions * 5 output tokens
    assert res["config"]["judge_model"] == judge.JUDGE_MODEL


def test_missing_fns_are_built_from_pinned_default(monkeypatch):
    """A real run (not recall_only) with no injected fns builds the answer/judge
    callables from the pinned default models — and the built model matches what
    the config records. (make_* patched to stubs so no anthropic / API key.)"""
    import eval.judge as judge_mod
    import eval.llm as llm_mod

    def fake_make_answer(model=None):
        assert model == llm_mod.ANSWER_MODEL  # None resolved to the pin
        return _echo_answer_fn

    def fake_make_judge(model=None, judge_prompt="repo"):
        assert model == judge_mod.JUDGE_MODEL
        assert judge_prompt == "repo"  # benchmark "t" has no paper default
        return _lenient_judge_fn

    monkeypatch.setattr(llm_mod, "make_answer_fn", fake_make_answer)
    monkeypatch.setattr(judge_mod, "make_judge_fn", fake_make_judge)

    res = run_benchmark([_conversation()], StubMemoryClient(), benchmark="t")
    assert res["config"]["judge_model"] == judge_mod.JUDGE_MODEL
    assert res["config"]["answer_model"] == llm_mod.ANSWER_MODEL
    assert res["summary"]["judge_accuracy"] == 1.0


# --- judge unit tests -------------------------------------------------------

def test_parse_verdict():
    assert judge.parse_verdict("CORRECT") is True
    assert judge.parse_verdict("INCORRECT") is False
    assert judge.parse_verdict("the answer is CORRECT") is True
    assert judge.parse_verdict("unparseable blah") is False  # conservative


def test_build_judge_prompt_abstention_note():
    p = judge.build_judge_prompt("q", "g", "a", abstention=True)
    assert "NOT answerable" in p
    assert judge.build_judge_prompt("q", "g", "a", abstention=False).count("NOT answerable") == 0


def test_judge_accuracy():
    assert judge.judge_accuracy([True, True, False, False]) == 0.5
    assert judge.judge_accuracy([]) == 0.0


def test_stub_memory_isolation():
    m = StubMemoryClient()
    m.store_facts_verbatim("p1", ["alpha beta"])
    m.store_facts_verbatim("p2", ["gamma delta"])
    assert m.recall_facts("p1", "alpha", k=5)  # p1 sees its fact
    assert m.recall_facts("p2", "alpha", k=5) == []  # p2 does not
