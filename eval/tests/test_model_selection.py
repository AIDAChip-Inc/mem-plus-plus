"""Selectable judge/answer model: name->id resolution, None => pinned default,
and the chosen models landing in the recorded config.

All API-free — the judge/answer callables are injected stubs, so no anthropic
client is constructed. The one test that actually builds a real client is gated
on ANTHROPIC_API_KEY and skips cleanly without one.
"""
from __future__ import annotations

import os

import pytest

from eval import judge as judge_mod
from eval import llm as llm_mod
from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import run_benchmark
from eval.llm import ANSWER_MODEL, LLMReply, resolve_model
from eval.judge import JUDGE_MODEL
from eval.memory_client import StubMemoryClient


def _conversation() -> Conversation:
    turns = [Turn(turn_id="t1", speaker="Alice", text="my favorite color is blue")]
    qa = [QAItem(qa_id="q1", question="what is Alice's favorite color?",
                 gold_answers=["blue"], category="single_hop", gold_turn_ids={"t1"})]
    return Conversation(conversation_id="c0", turns=turns, qa=qa)


def _stub_answer(prompt: str) -> LLMReply:
    return LLMReply(text="blue", input_tokens=5, output_tokens=1)


def _stub_judge(prompt: str) -> str:
    return "CORRECT"


# --- name -> id resolution --------------------------------------------------

def test_resolve_friendly_names():
    assert resolve_model("haiku", JUDGE_MODEL) == "claude-haiku-4-5-20251001"
    assert resolve_model("sonnet", JUDGE_MODEL) == "claude-sonnet-5"
    assert resolve_model("opus", JUDGE_MODEL) == "claude-opus-4-8"


def test_resolve_is_case_insensitive_and_trims():
    assert resolve_model("  OPUS ", JUDGE_MODEL) == "claude-opus-4-8"
    assert resolve_model("Sonnet", JUDGE_MODEL) == "claude-sonnet-5"


def test_resolve_none_uses_default_pin():
    # None reproduces the frozen baseline for BOTH sides.
    assert resolve_model(None, JUDGE_MODEL) == JUDGE_MODEL == "claude-haiku-4-5-20251001"
    assert resolve_model(None, ANSWER_MODEL) == ANSWER_MODEL
    # "haiku" resolves to the same pin, so the friendly name == the default.
    assert resolve_model("haiku", JUDGE_MODEL) == JUDGE_MODEL


def test_resolve_raw_id_passthrough():
    assert resolve_model("claude-opus-4-8", JUDGE_MODEL) == "claude-opus-4-8"
    assert resolve_model("some-future-model-id", JUDGE_MODEL) == "some-future-model-id"


# --- config recording -------------------------------------------------------

def test_none_records_pinned_default():
    res = run_benchmark([_conversation()], StubMemoryClient(),
                        answer_fn=_stub_answer, judge_fn=_stub_judge, benchmark="t")
    cfg = res["config"]
    assert cfg["judge_model"] == JUDGE_MODEL
    assert cfg["answer_model"] == ANSWER_MODEL
    assert cfg["judge_model_pinned_default"] == JUDGE_MODEL
    assert cfg["prompt_versions_pinned"] is True
    # prompt version stays fixed while the model is user-selected.
    assert cfg["judge_prompt_version"] == judge_mod.JUDGE_PROMPT_VERSION


def test_chosen_models_land_in_config():
    res = run_benchmark([_conversation()], StubMemoryClient(),
                        answer_fn=_stub_answer, judge_fn=_stub_judge,
                        judge_model="sonnet", answer_model="opus", benchmark="t")
    cfg = res["config"]
    assert cfg["judge_model"] == "claude-sonnet-5"
    assert cfg["answer_model"] == "claude-opus-4-8"
    # the pinned-default fields still record the frozen baseline for comparison.
    assert cfg["judge_model_pinned_default"] == JUDGE_MODEL
    # prompt version unchanged — only the MODEL swapped.
    assert cfg["judge_prompt_version"] == judge_mod.JUDGE_PROMPT_VERSION


def test_raw_id_recorded_verbatim():
    res = run_benchmark([_conversation()], StubMemoryClient(),
                        answer_fn=_stub_answer, judge_fn=_stub_judge,
                        judge_model="claude-opus-4-8", benchmark="t")
    assert res["config"]["judge_model"] == "claude-opus-4-8"


def test_recall_only_records_no_models():
    res = run_benchmark([_conversation()], StubMemoryClient(),
                        recall_only=True, judge_model="opus", benchmark="t")
    cfg = res["config"]
    assert cfg["judge_model"] is None
    assert cfg["answer_model"] is None


# --- gated real-client build (no tokens spent; skips without a key) ---------

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                    reason="needs ANTHROPIC_API_KEY to construct the anthropic client")
def test_make_fns_build_for_selected_models():
    assert callable(judge_mod.make_judge_fn("opus"))
    assert callable(llm_mod.make_answer_fn("sonnet"))
