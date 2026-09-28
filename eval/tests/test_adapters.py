"""Adapter tests: native benchmark JSON → common Conversation/QAItem format."""
from __future__ import annotations

import json

from eval.adapters import locomo, longbench, longmemeval, membench

LOCOMO_SAMPLE = [{
    "conversation": {
        "session_1_date_time": "1:56 pm on 7 May, 2023",
        "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "I love hiking"}],
        "session_2_date_time": "2:00 pm on 8 May, 2023",
        "session_2": [{"speaker": "Bob", "dia_id": "D2:1", "text": "Nice"}],
    },
    "qa": [
        {"question": "What does Alice love?", "answer": "hiking", "category": 4, "evidence": ["D1:1"]},
        {"question": "Trap?", "adversarial_answer": "nope", "category": 5, "evidence": []},
    ],
}]


def test_locomo_adapter(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_SAMPLE))
    convs = locomo.load(p)
    assert len(convs) == 1
    c = convs[0]
    assert [t.turn_id for t in c.turns] == ["D1:1", "D2:1"]
    assert c.turns[0].occurred_at is not None  # date parsed
    assert len(c.qa) == 2
    q0, q1 = c.qa
    assert q0.category == "single_hop" and q0.gold_turn_ids == {"D1:1"} and not q0.abstention
    assert q1.category == "adversarial" and q1.abstention
    assert q1.gold_answers == [locomo.ADVERSARIAL_GOLD]


LME_SAMPLE = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "fav color?",
        "answer": "blue",
        "haystack_sessions": [[
            {"role": "user", "content": "my fav color is blue", "has_answer": True},
            {"role": "assistant", "content": "noted"},
        ]],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/07 (Sun) 13:56"],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "q2_abs",
        "question_type": "knowledge-update",
        "question": "unanswerable?",
        "answer": "n/a",
        "haystack_sessions": [[{"role": "user", "content": "hello", "has_answer": False}]],
        "haystack_session_ids": ["s9"],
        "haystack_dates": ["2023/05/07"],
        "answer_session_ids": [],
    },
]


def test_longmemeval_adapter(tmp_path):
    p = tmp_path / "longmemeval_s.json"
    p.write_text(json.dumps(LME_SAMPLE))
    convs = longmemeval.load(p)
    assert len(convs) == 2
    c0 = convs[0]
    assert [t.turn_id for t in c0.turns] == ["s1:0", "s1:1"]
    assert c0.qa[0].gold_turn_ids == {"s1:0"}  # has_answer flag on turn 0
    assert c0.qa[0].category == "single_session_user" and not c0.qa[0].abstention
    assert convs[1].qa[0].abstention  # _abs suffix


def test_longbench_adapter(tmp_path):
    p = tmp_path / "qasper.jsonl"
    p.write_text(json.dumps({
        "input": "What is X?",
        "context": "Para one.\n\nPara two.",
        "answers": ["the answer"],
        "length": 123,
    }) + "\n")
    convs = longbench.load(p)
    assert len(convs) == 1
    c = convs[0]
    assert len(c.turns) == 2  # two paragraphs → two chunks
    assert c.qa[0].category == "qasper"
    assert c.qa[0].gold_answers == ["the answer"]
    assert c.qa[0].gold_turn_ids == set()  # no turn-level evidence in LongBench


def test_longbench_adapter_directory(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps(
        {"input": "q", "context": "one", "answers": ["x"]}) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(
        {"input": "q2", "context": "two", "answers": ["y"]}) + "\n")
    convs = longbench.load(tmp_path)
    assert {c.qa[0].category for c in convs} == {"a", "b"}


# ThirdAgent (observation): flat message_list, flat-int target_step_id.
MEMBENCH_THIRD = {
    "roles": [{
        "tid": 0,
        "message_list": [
            {"mid": 0, "message": "Clara has a Master's degree.",
             "time": "'2024-10-01 08:00' Tuesday", "place": "Boston, MA"},
            {"mid": 1, "message": "Maya has a Bachelor's degree.",
             "time": "'2024-10-01 09:00' Tuesday", "place": "Boston, MA"},
            {"mid": 2, "message": "The weather was nice.",
             "time": "'2024-10-01 10:00' Tuesday", "place": "Boston, MA"},
        ],
        "QA": {"qid": 0, "question": "Who is more educated?",
               "answer": "Clara", "target_step_id": [0, 1],
               "choices": {"A": "Maya", "B": "Same", "C": "Neither", "D": "Clara"},
               "ground_truth": "D", "time": "'2024-10-06 10:00' Sunday"},
    }],
}

# FirstAgent (participation): nested sessions, [global-mid, outer] target pairs.
MEMBENCH_FIRST = {
    "movie": [{
        "tid": 7,
        "message_list": [
            [{"mid": 0, "user": "I love The Godfather.", "assistant": "Great classic!",
              "time": "'2024-10-01 08:00' Tuesday", "place": "Boston, MA"}],
            [{"mid": 1, "user": "Recommend me a drama.", "assistant": "Try The Godfather.",
              "time": "'2024-10-02 08:00' Wednesday", "place": "Boston, MA"}],
        ],
        "QA": {"qid": 0, "question": "What genre do I prefer?",
               "answer": "Drama", "target_step_id": [[0, 0], [1, 1]],
               "choices": {"A": "Musical", "B": "Drama", "C": "Horror", "D": "Kids"},
               "ground_truth": "B", "time": "'2024-10-03 08:00' Thursday"},
    }],
}


def test_membench_third_agent_flat(tmp_path):
    d = tmp_path / "ThirdAgent"
    d.mkdir()
    (d / "comparative.json").write_text(json.dumps(MEMBENCH_THIRD))
    convs = membench.load(d / "comparative.json")
    assert len(convs) == 1
    c = convs[0]
    # mid is the turn provenance id; three flat statements → three turns.
    assert [t.turn_id for t in c.turns] == ["0", "1", "2"]
    assert c.turns[0].occurred_at is not None  # time parsed
    q = c.qa[0]
    assert q.category == "ThirdAgent:comparative"
    assert q.gold_turn_ids == {"0", "1"}  # flat-int target ids
    assert not q.abstention
    assert "Options:" in q.question and "D. Clara" in q.question  # choices embedded
    assert q.primary_gold == "D. Clara"  # letter + text for the judge
    assert "Clara" in q.gold_answers and "D" in q.gold_answers


def test_membench_first_agent_nested(tmp_path):
    d = tmp_path / "FirstAgent"
    d.mkdir()
    (d / "highlevel.json").write_text(json.dumps(MEMBENCH_FIRST))
    convs = membench.load(d)  # directory recursion
    assert len(convs) == 1
    c = convs[0]
    assert c.conversation_id == "membench_FirstAgent_highlevel_movie_7"
    # nested sessions flatten by global mid; user+assistant render into one turn.
    assert [t.turn_id for t in c.turns] == ["0", "1"]
    assert "User: I love The Godfather." in c.turns[0].text
    assert "Assistant: Great classic!" in c.turns[0].text
    q = c.qa[0]
    assert q.category == "FirstAgent:highlevel"
    # [mid, outer] pairs → gold ids are the global mids (pair[0]), not positions.
    assert q.gold_turn_ids == {"0", "1"}
    assert q.primary_gold == "B. Drama"
