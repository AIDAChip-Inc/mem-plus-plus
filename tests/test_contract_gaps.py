"""Contract/seam regression guards for ``store_facts_verbatim``.

Burhan found three defects here (A: dict form rejected; B: ISO ``occurred_at``
string crashed; C: the eval harness fed bare strings and the engine raised
``too many values to unpack``). They were pinned ``xfail(strict=True)`` at
discovery, then fixed at integration by a single normalization seam in
``memory/recall.py::store_facts_verbatim`` (accepts str / dict / 3-tuple and
coerces ISO ``occurred_at``, faithful to production ``team_memory.py`` and
INTERFACE_CONTRACT §3). These now assert the fixed behavior — they must stay green.

See E2E_STATUS.md for the full writeup and the DB-gated reproduction commands.
"""
from __future__ import annotations

from datetime import UTC, datetime

from memory import recall


def test_store_facts_verbatim_accepts_canonical_dict_form(sqlite_public_api):
    res = recall.store_facts_verbatim(
        "p", [{"summary": "a durable fact", "tags": ["t"], "occurred_at": None}]
    )
    assert res["written"] == 1
    rows = recall.recall_facts("p", "durable fact", k=5)
    assert any("durable fact" in r["summary"] for r in rows)


def test_store_facts_verbatim_parses_iso_string_occurred_at(sqlite_public_api):
    res = recall.store_facts_verbatim("p", [("dated fact", [], "2026-07-20")])
    assert res["written"] == 1
    rows = recall.recall_facts("p", "dated fact", k=5)
    assert rows[0]["occurred_at"] == "2026-07-20"


def test_harness_string_facts_reach_engine_without_crashing(sqlite_public_api):
    # This is exactly what eval/harness.py passes per conversation.
    res = recall.store_facts_verbatim(
        "researcher:locomo_0",
        ["[2023-05-15] Alice: I finally booked my trip to Kyoto"],
    )
    assert res["written"] == 1


def test_store_facts_verbatim_datetime_tuple_is_the_working_form(sqlite_public_api):
    """Control (GREEN): the de-facto supported form — a 3-tuple whose occurred_at
    is a real datetime — works. This is what tests_engine and the session-close
    path use; it pins the one shape callers can rely on today."""
    res = recall.store_facts_verbatim(
        "p", [("a working fact", ["ok"], datetime(2026, 7, 20, tzinfo=UTC))]
    )
    assert res["written"] == 1
    rows = recall.recall_facts("p", "working fact", k=5)
    assert rows[0]["occurred_at"] == "2026-07-20"


def test_recall_facts_as_of_bound_reaches_only_dated_rows(sqlite_public_api):
    # What eval/harness.py stores for a LongMemEval question with a question_date:
    # dict facts whose occurred_at is the session's ISO datetime.
    recall.store_facts_verbatim("p", [
        {"summary": "[2023-05-20] user: adopted a cat", "occurred_at": "2023-05-20T02:21:00+00:00"},
        {"summary": "[2023-06-05] user: adopted a dog", "occurred_at": "2023-06-05T10:00:00+00:00"},
        "[?] user: adopted a bird",  # undated: fails the as-of condition
    ])
    theta = datetime(2023, 5, 30, 23, 40, tzinfo=UTC)
    rows = recall.recall_facts("p", "adopted", k=10, occurred_before=theta)
    assert [r["summary"] for r in rows] == ["[2023-05-20] user: adopted a cat"]
    assert len(recall.recall_facts("p", "adopted", k=10)) == 3  # no bound: all rows
