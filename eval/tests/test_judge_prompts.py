"""Judge-prompt selection, question filters and recorded recall params.

No network: the OpenAI client is replaced by a fake module, and every run uses
the stub memory backend.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest

from eval import judge, run_locomo, run_longmemeval
from eval.adapters.base import Conversation, QAItem, Turn
from eval.harness import exclude_abstention_items, run_benchmark
from eval.llm import LLMReply
from eval.memory_client import StubMemoryClient


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- upstream prompts are byte-exact ------------------------------------------

def test_mem0_prompt_is_verbatim():
    # sha256 of ACCURACY_PROMPT in mem0ai/mem0@aae5989 evaluation/metrics/llm_judge.py
    assert _sha(judge.MEM0_ACCURACY_PROMPT) == (
        "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0")


@pytest.mark.parametrize(("name", "digest"), [
    ("_LME_DEFAULT", "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc"),
    ("_LME_TEMPORAL", "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273"),
    ("_LME_KNOWLEDGE_UPDATE", "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac"),
    ("_LME_PREFERENCE", "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b"),
    ("_LME_ABSTENTION", "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b"),
])
def test_longmemeval_templates_are_verbatim(name, digest):
    # sha256 of each template string in LongMemEval@d6dc8b5 get_anscheck_prompt
    assert _sha(getattr(judge, name)) == digest


# --- default per benchmark -----------------------------------------------------

def test_default_judge_prompt_matches_paper():
    assert judge.default_judge_prompt("locomo") == "mem0"
    assert judge.default_judge_prompt("longmemeval") == "longmemeval"
    assert judge.default_judge_prompt("longbench") == "repo"
    assert judge.default_judge_prompt("membench") == "repo"
    assert set(judge.JUDGE_PROMPTS) == {"repo", "mem0", "longmemeval"}


# --- prompt building -----------------------------------------------------------

def test_build_mem0_prompt():
    p = judge.build_judge_prompt("Q?", "gold", "gen", judge_prompt="mem0")
    assert "Question: Q?\nGold answer: gold\nGenerated answer: gen\n" in p
    assert "be generous" in p and '"label"' in p


@pytest.mark.parametrize(("category", "marker"), [
    ("single_session_user", "contains all the intermediate steps"),
    ("multi_session", "contains all the intermediate steps"),
    ("temporal_reasoning", "off-by-one"),
    ("knowledge-update", "updated answer"),
    ("single_session_preference", "Rubric: g"),
])
def test_build_longmemeval_prompt_per_type(category, marker):
    p = judge.build_judge_prompt("q", "g", "r", judge_prompt="longmemeval", category=category)
    assert marker in p
    assert p.endswith("Answer yes or no only.")


def test_build_longmemeval_abstention_prompt():
    p = judge.build_judge_prompt("q", "why", "r", judge_prompt="longmemeval",
                                 category="multi_session", abstention=True)
    assert p.startswith("I will give you an unanswerable question")
    assert "Explanation: why" in p


def test_longmemeval_prompt_rejects_non_lme_category():
    with pytest.raises(ValueError, match="question type"):
        judge.build_judge_prompt("q", "g", "r", judge_prompt="longmemeval", category="single_hop")


def test_unknown_prompt_rejected():
    with pytest.raises(ValueError, match="unknown judge prompt"):
        judge.build_judge_prompt("q", "g", "r", judge_prompt="nope")


# --- verdict parsing -----------------------------------------------------------

@pytest.mark.parametrize(("reply", "expected"), [
    ('{"label": "CORRECT"}', True),
    ('{"label": "WRONG"}', False),
    ('```json\n{"label": "CORRECT"}\n```', True),
    ('Reasoning first. {"label": "CORRECT"}', True),
    ("CORRECT", False),  # not JSON, no label field -> fail closed
    ("", False),
])
def test_parse_mem0(reply, expected):
    assert judge.parse_verdict(reply, "mem0") is expected


@pytest.mark.parametrize(("reply", "expected"), [
    ("yes", True), ("Yes.", True), ("no", False), ("", False),
])
def test_parse_longmemeval(reply, expected):
    assert judge.parse_verdict(reply, "longmemeval") is expected


def test_parse_repo_unchanged():
    assert judge.parse_verdict("CORRECT") is True
    assert judge.parse_verdict("INCORRECT") is False


# --- make_judge_fn call parameters (fake OpenAI client) ---------------------------

@pytest.fixture()
def fake_openai(monkeypatch):
    calls: list[dict] = []

    class _Completions:
        def create(self, **kw):
            calls.append(kw)
            msg = types.SimpleNamespace(content='{"label": "CORRECT"}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class OpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    return calls


def test_make_judge_fn_mem0_uses_json_mode_temperature_zero(fake_openai):
    fn = judge.make_judge_fn("gpt-4o-mini", judge_prompt="mem0")
    assert fn("prompt") == '{"label": "CORRECT"}'
    kw = fake_openai[-1]
    assert kw["model"] == "gpt-4o-mini"
    assert kw["temperature"] == 0
    assert kw["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("prompt_name", ["repo", "longmemeval"])
def test_make_judge_fn_other_prompts_no_json_mode(fake_openai, prompt_name):
    judge.make_judge_fn("openai:gpt-4o-mini", judge_prompt=prompt_name)("p")
    kw = fake_openai[-1]
    assert kw["temperature"] == 0
    assert "response_format" not in kw


# --- harness: prompt routing + recorded config ----------------------------------

def _conv(abstention_item: bool = True) -> Conversation:
    turns = [Turn(turn_id="t1", speaker="Alice", text="my favorite color is blue")]
    qa = [QAItem(qa_id="q1", question="favorite color?", gold_answers=["blue"],
                 category="single_hop", gold_turn_ids={"t1"})]
    if abstention_item:
        qa.append(QAItem(qa_id="q2", question="pet name?", gold_answers=["n/a"],
                         category="adversarial", abstention=True))
    return Conversation(conversation_id="c0", turns=turns, qa=qa)


def _answer(prompt: str) -> LLMReply:
    return LLMReply(text="blue", input_tokens=1, output_tokens=1)


def test_harness_locomo_defaults_to_mem0_prompt():
    seen: list[str] = []

    def judge_fn(prompt: str) -> str:
        seen.append(prompt)
        return '{"label": "CORRECT"}'

    res = run_benchmark([_conv(False)], StubMemoryClient(), answer_fn=_answer,
                        judge_fn=judge_fn, benchmark="locomo", backend="stub",
                        count_tokens=False)
    assert "be generous" in seen[0]
    assert res["config"]["judge_prompt"] == "mem0"
    assert res["config"]["judge_prompt_version"] == judge.JUDGE_PROMPT_VERSIONS["mem0"]
    assert res["summary"]["judge_accuracy"] == 1.0


def test_harness_explicit_prompt_overrides_default():
    res = run_benchmark([_conv(False)], StubMemoryClient(), answer_fn=_answer,
                        judge_fn=lambda p: "CORRECT", benchmark="locomo",
                        judge_prompt="repo", backend="stub", count_tokens=False)
    assert res["config"]["judge_prompt"] == "repo"
    assert res["config"]["judge_prompt_version"] == judge.JUDGE_PROMPT_VERSION
    assert res["summary"]["judge_accuracy"] == 1.0


def test_recall_params_record_live_config(monkeypatch):
    import memory.config as mc

    monkeypatch.setattr(mc, "MEMORY_RRF_W_VECTOR", 4.0)
    monkeypatch.setattr(mc, "CANDIDATE_LIMIT", 50)
    monkeypatch.setattr(mc, "MEMORY_LEXICAL_OR", False)
    res = run_benchmark([_conv(False)], StubMemoryClient(), recall_only=True,
                        benchmark="t", backend="stub", count_tokens=False)
    rp = res["config"]["recall_params"]
    assert rp["weights"] == [1.0, 1.0, 4.0]
    assert rp["candidate_limit"] == 50
    assert rp["lexical_or"] is False


def test_exclude_abstention_items_drops_empty_conversations():
    only_abs = Conversation("c1", [], [QAItem("x", "q", ["n/a"], "adversarial", abstention=True)])
    kept, rec = exclude_abstention_items([_conv(True), only_abs], label="adversarial")
    assert [c.conversation_id for c in kept] == ["c0"]
    assert [q.qa_id for q in kept[0].qa] == ["q1"]
    assert rec == {"exclude": "adversarial", "n_questions_before": 3,
                   "n_excluded": 2, "n_questions_after": 1}


# --- CLI: flags, defaults, recorded n --------------------------------------------

LOCOMO_SAMPLE = [{
    "conversation": {
        "session_1_date_time": "1:56 pm on 7 May, 2023",
        "session_1": [
            {"speaker": "Alice", "dia_id": "D1:1", "text": "my favorite color is blue"},
        ],
    },
    "qa": [
        {"question": "what is Alice's favorite color?", "answer": "blue",
         "category": 4, "evidence": ["D1:1"]},
        {"question": "what is Bob's favorite color?", "adversarial_answer": "red",
         "category": 5, "evidence": []},
    ],
}]

LME_SAMPLE = [
    {
        "question_id": "q1", "question_type": "single-session-user",
        "question": "fav color?", "answer": "blue",
        "haystack_sessions": [[{"role": "user", "content": "my fav color is blue",
                                "has_answer": True}]],
        "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/07"],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "q2_abs", "question_type": "knowledge-update",
        "question": "unanswerable?", "answer": "never mentioned",
        "haystack_sessions": [[{"role": "user", "content": "hello"}]],
        "haystack_session_ids": ["s9"], "haystack_dates": ["2023/05/07"],
        "answer_session_ids": [],
    },
]


def _run(module, sample, tmp_path, *extra):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(sample))
    out = tmp_path / "res.json"
    assert module.main(["--stub", "--no-count-tokens", "--data", str(data),
                        "--out", str(out), *extra]) == 0
    return json.loads(out.read_text())


def test_locomo_cli_defaults(tmp_path):
    cfg = _run(run_locomo, LOCOMO_SAMPLE, tmp_path)["config"]
    assert cfg["judge_prompt"] == "mem0"
    assert cfg["n_questions"] == 1
    assert cfg["question_filter"] == {"exclude": "adversarial", "n_questions_before": 2,
                                      "n_excluded": 1, "n_questions_after": 1}


def test_locomo_cli_keep_adversarial_and_repo_judge(tmp_path):
    res = _run(run_locomo, LOCOMO_SAMPLE, tmp_path,
               "--no-exclude-adversarial", "--judge-prompt", "repo")
    cfg = res["config"]
    assert cfg["judge_prompt"] == "repo"
    assert cfg["n_questions"] == 2
    assert cfg["question_filter"]["exclude"] is None
    assert cfg["question_filter"]["n_excluded"] == 0


@pytest.mark.parametrize("prompt_name", ["repo", "mem0", "longmemeval"])
def test_stub_judge_speaks_each_prompt_format(prompt_name):
    # --stub dry-runs must emit a verdict the selected prompt's parser accepts,
    # and read the real gold line (not mem0's in-prompt example).
    from eval._runner import _stub_judge_fn

    fn = _stub_judge_fn(prompt_name)

    def verdict(gen: str) -> bool:
        prompt = judge.build_judge_prompt("q", "blue car", gen, judge_prompt=prompt_name,
                                          category="single_session_user")
        return judge.parse_verdict(fn(prompt), prompt_name)

    assert verdict("blue car") is True
    assert verdict("a shell necklace") is False


def test_longmemeval_cli_defaults(tmp_path):
    res = _run(run_longmemeval, LME_SAMPLE, tmp_path)
    cfg = res["config"]
    assert cfg["judge_prompt"] == "longmemeval"
    assert cfg["judge_prompt_version"] == judge.JUDGE_PROMPT_VERSIONS["longmemeval"]
    assert cfg["n_questions"] == 1 and cfg["n_conversations"] == 1
    assert cfg["question_filter"]["exclude"] == "abstention"


def test_longmemeval_cli_keep_abstention(tmp_path):
    cfg = _run(run_longmemeval, LME_SAMPLE, tmp_path, "--no-exclude-abstention")["config"]
    assert cfg["n_questions"] == 2


def test_filter_flags_only_on_their_benchmark(tmp_path):
    with pytest.raises(SystemExit):
        run_locomo.main(["--exclude-abstention"])
    with pytest.raises(SystemExit):
        run_longmemeval.main(["--exclude-adversarial"])
