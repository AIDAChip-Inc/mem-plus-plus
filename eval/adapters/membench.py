"""MemBench adapter: MemData/<Agent>/<qatype>.json → common format.

Source: ``import-myself/Membench`` (MemBench, Tan et al., ACL 2025 Findings;
arXiv 2506.21605), fetched by ``datasets/fetch.py membench`` into
``datasets/membench/{FirstAgent,ThirdAgent}/<qatype>.json``. MemBench evaluates
LLM-agent memory across two levels (factual / reflective) and two interactive
scenarios (participation = FirstAgent dialogue sessions; observation =
ThirdAgent statement streams).

Native file shape: ``{scenario: [trajectory, ...]}``. Each trajectory carries a
``tid``, a ``message_list``, and one ``QA``:

  * ``message_list`` is EITHER a flat list of message dicts (ThirdAgent factual:
    ``{mid, message, time, place, ...}``) OR a list of sub-lists — sessions
    (FirstAgent: ``{mid, user, assistant, time, place}``) or grouped statements
    (ThirdAgent reflective: ``{mid, message, time, place}``). In every shape the
    ``mid`` is GLOBAL and contiguous across sub-lists, so it is the single turn
    provenance id.
  * ``QA`` is multiple-choice: ``question``, ``choices`` (A–D; values are strings
    or string lists), ``ground_truth`` (the correct letter), ``answer`` (the gold
    text), and ``target_step_id`` — the gold evidence. A flat entry is a ``mid``;
    a ``[mid, outer_index]`` pair references the same global ``mid`` (verified:
    ``target[0]`` is the mid, never a within-sublist position).

Protocol note (own-baseline): MemBench's native metric is exact letter-match
accuracy. This harness measures under ITS OWN frozen protocol — the choices are
rendered into the question and the answer is scored by token-F1 + LLM-judge
against the gold option (letter + text) — so MemBench numbers are directly
comparable to LoCoMo/LongMemEval/LongBench here, and NOT to the paper's leaderboard.

No-leakage is structural: ingestion consumes only ``turns``, never ``qa``.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .base import Conversation, QAItem, Turn

# Time strings look like "'2024-10-01 08:00' Tuesday" — pull the ISO-ish core.
_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})")


def _parse_dt(raw) -> datetime | None:
    if not isinstance(raw, str):
        return None
    m = _TS.search(raw)
    if not m:
        return None
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def _render_choice(value) -> str:
    """A choice / answer value is a string or a list of strings."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _turn_from_message(msg: dict) -> Turn:
    """One message dict → one ingestable Turn, keyed by its global ``mid``.

    ThirdAgent messages carry ``message``; FirstAgent turns carry ``user`` /
    ``assistant`` (a dialogue exchange rendered as one turn so gold evidence,
    which points at a single ``mid``, maps 1:1)."""
    occurred = _parse_dt(msg.get("time"))
    if "message" in msg:  # observation: a single statement
        return Turn(turn_id=str(msg.get("mid")), speaker="user",
                    text=str(msg["message"]).strip(), occurred_at=occurred)
    # participation: a user/assistant exchange (key spellings vary across files)
    user = msg.get("user", msg.get("user_message", ""))
    assistant = msg.get("assistant", msg.get("assistant_message", ""))
    text = f"User: {str(user).strip()}\nAssistant: {str(assistant).strip()}"
    return Turn(turn_id=str(msg.get("mid")), speaker="dialogue", text=text, occurred_at=occurred)


def _iter_messages(message_list):
    """Yield message dicts from a flat list OR a list of sub-lists (sessions /
    grouped statements). ``mid`` is global, so nesting only affects iteration."""
    for entry in message_list or []:
        if isinstance(entry, list):
            yield from entry
        else:
            yield entry


def _gold_turn_ids(target_step_id) -> set[str]:
    """Gold evidence ids: a flat entry is a ``mid``; a ``[mid, outer]`` pair's
    first element is the same global ``mid``."""
    ids: set[str] = set()
    for entry in target_step_id or []:
        mid = entry[0] if isinstance(entry, (list, tuple)) else entry
        ids.add(str(mid))
    return ids


def _render_question(qa: dict) -> str:
    """Embed the multiple-choice options into the question so the answerer sees
    them (the harness passes only the question string to the answer prompt)."""
    lines = [str(qa.get("question", "")), "", "Options:"]
    for letter, value in (qa.get("choices") or {}).items():
        lines.append(f"{letter}. {_render_choice(value)}")
    lines += ["", "Answer with the text of the single best option."]
    return "\n".join(lines)


def _gold_answers(qa: dict) -> list[str]:
    """Acceptable gold answers, most-informative first (``primary_gold`` = the
    letter+text form the LLM-judge sees): the correct option as ``<letter>. <text>``,
    its bare text, the dataset's ``answer`` text, and the bare letter — deduped."""
    letter = str(qa.get("ground_truth", "")).strip()
    choice_text = _render_choice((qa.get("choices") or {}).get(letter))
    answer_text = _render_choice(qa.get("answer"))
    labeled = f"{letter}. {choice_text}" if letter and choice_text else (choice_text or letter)
    candidates = [labeled, choice_text, answer_text, letter]
    golds: list[str] = []
    for c in candidates:
        if c and c not in golds:
            golds.append(c)
    return golds or [letter]


def _to_conversation(traj: dict, category: str, conv_id: str) -> Conversation:
    turns = [_turn_from_message(m) for m in _iter_messages(traj.get("message_list"))]
    qa = traj.get("QA") or {}
    item = QAItem(
        qa_id=f"{conv_id}:q{qa.get('qid', 0)}",
        question=_render_question(qa),
        gold_answers=_gold_answers(qa),
        category=category,
        gold_turn_ids=_gold_turn_ids(qa.get("target_step_id")),
        abstention=False,  # every MemBench question has a ground-truth option
    )
    return Conversation(conversation_id=conv_id, turns=turns, qa=[item])


def load(path: str | Path) -> list[Conversation]:
    """Load MemBench data into the common format. ``path`` may be a single
    ``<qatype>.json``, an agent dir, or the ``datasets/membench`` root (recursed).
    The category is ``<agent>:<qatype>`` (parent folder + file stem) so the two
    agents' same-named task files (e.g. ``highlevel``) stay distinct."""
    path = Path(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    conversations: list[Conversation] = []
    for f in files:
        category = f"{f.parent.name}:{f.stem}"
        data = json.loads(f.read_text())
        for scenario, trajs in data.items():
            for i, traj in enumerate(trajs):
                conv_id = f"membench_{f.parent.name}_{f.stem}_{scenario}_{traj.get('tid', i)}"
                conversations.append(_to_conversation(traj, category, conv_id))
    return conversations
