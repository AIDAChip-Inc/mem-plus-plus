"""LongMemEval adapter: longmemeval_{s,m,oracle}.json → common format.

Source: HF ``xiaowu0162/longmemeval-cleaned`` (fetched by
``datasets/fetch.py longmemeval``). Each instance is one (haystack, question)
pair — modeled here as a one-instance Conversation so retrieval gold is
session-scoped.

Per-instance fields: ``question_id`` (``_abs`` suffix → abstention),
``question_type``, ``question``, ``answer``, ``haystack_sessions`` (list of
sessions, each a list of ``{role, content, has_answer}`` turns),
``haystack_session_ids``, ``haystack_dates``, ``answer_session_ids``.

Turn provenance id = ``<session_id>:<turn_index>``. Gold evidence = every turn
in an answer session that is flagged ``has_answer`` (falls back to all turns of
the answer sessions when no per-turn flag is present).
"""
from __future__ import annotations

import codecs
import json
from collections.abc import Container, Iterator
from datetime import UTC, datetime
from pathlib import Path

from .base import Conversation, QAItem, Turn

_DT_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d")
# Read granularity for the streaming parser. 4 MiB is large enough that a single
# instance (max measured 616 turns, ~1.3 MB) usually completes inside one or two
# chunks, and small enough that the resident buffer never approaches the file.
_CHUNK_BYTES = 4 << 20


def _parse_dt(raw) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = " ".join(raw.strip().split())
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _to_conversation(item: dict, idx: int) -> Conversation:
    qid = str(item.get("question_id", f"q{idx}"))
    abstention = qid.endswith("_abs")
    answer_sids = set(item.get("answer_session_ids") or [])

    sessions = item.get("haystack_sessions") or []
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []

    turns: list[Turn] = []
    gold_turn_ids: set[str] = set()
    for sess, sid, date in zip(sessions, session_ids, dates, strict=False):
        occurred = _parse_dt(date)
        # When a session has no per-turn has_answer flags, treat every turn of
        # an answer session as gold evidence (session-level gold).
        flagged = any(t.get("has_answer") for t in sess)
        for ti, t in enumerate(sess):
            body = (t.get("content") or "").strip()
            if not body:
                continue
            turn_id = f"{sid}:{ti}"
            turns.append(Turn(turn_id=turn_id, speaker=str(t.get("role", "?")),
                              text=body, occurred_at=occurred))
            if sid in answer_sids and (t.get("has_answer") or not flagged):
                gold_turn_ids.add(turn_id)

    qa = [QAItem(
        qa_id=qid,
        question=str(item.get("question", "")),
        gold_answers=[str(item.get("answer", ""))],
        category=str(item.get("question_type", "unknown")).replace("-", "_"),
        gold_turn_ids=gold_turn_ids,
        abstention=abstention,
    )]
    return Conversation(conversation_id=f"lme_{qid}", turns=turns, qa=qa)


def iter_raw(path: str | Path) -> Iterator[dict]:
    """Yield the top-level objects of a longmemeval_*.json array, one at a time.

    WHY NOT ``json.loads(Path(path).read_text())``. That is what ``load()`` used
    to do, and it cannot read ``longmemeval_s_cleaned.json`` (277 MB) reliably on
    Windows for two reasons, both of which this function removes:

      * MEMORY. ``read_text`` materialises a 277 MB ``str`` AND ``json.loads``
        then builds the entire decoded tree (~2.4 GB of dicts/lists/strs for this
        file) — both resident at once, before a single ``Conversation`` exists.
        Streaming keeps one instance's tree alive at a time, so the peak is the
        output list plus ~5 MB of buffer.
      * ENCODING. ``read_text`` opens in TEXT mode: it applies universal-newline
        translation (measured: 130,211 CRLF pairs rewritten, i.e. a full extra
        pass over 277 MB) and, with ``encoding="utf-8"``, a UTF-8 BOM would
        survive into the string and make ``json.loads`` raise
        ``JSONDecodeError: Expecting value: line 1 column 1``. Reading BINARY and
        decoding with an incremental ``utf-8-sig`` decoder fixes both, and never
        splits a multi-byte character across a chunk boundary.

    Falls back to a whole-file parse if the payload is not a top-level array (the
    ``_oracle``/``_m`` variants are arrays too, so this is defence only).
    """
    dec = json.JSONDecoder()
    inc = codecs.getincrementaldecoder("utf-8-sig")()
    with open(path, "rb") as fh:
        buf = ""
        pos = 0                      # cursor into buf
        started = False
        while True:
            raw = fh.read(_CHUNK_BYTES)
            buf = buf[pos:] + inc.decode(raw, not raw)
            pos = 0
            if not started:
                lead = buf.lstrip()
                if not lead:
                    if not raw:
                        return
                    continue
                if lead[0] != "[":
                    # not an array — fall back to a single whole-file parse
                    yield from _whole_file_items(path)
                    return
                pos = buf.index("[") + 1
                started = True
            while True:
                # skip whitespace and the ',' separators between elements
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break
                try:
                    obj, pos = dec.raw_decode(buf, pos)
                except ValueError:
                    break            # incomplete element — pull another chunk
                yield obj
            if not raw:
                return


def _whole_file_items(path: str | Path) -> list:
    data = json.loads(Path(path).read_bytes().decode("utf-8-sig"))
    return list(data.values()) if isinstance(data, dict) else list(data)


def load(path: str | Path, question_ids: Container[str] | None = None) -> list[Conversation]:
    """Load a longmemeval_*.json file into the common format (one Conversation
    per question instance).

    ``question_ids`` (optional) keeps only the instances whose ``question_id`` is
    in the given container — used to run a stratified subset without building the
    other 400 Conversations. ``None`` (the default) loads every instance, exactly
    as before. Item indices are the FULL-FILE indices either way, so the ``idx``
    fallback id and any index-based subset recipe stay stable under filtering.
    """
    out: list[Conversation] = []
    for i, item in enumerate(iter_raw(path)):
        if question_ids is not None and str(item.get("question_id", f"q{i}")) not in question_ids:
            continue
        out.append(_to_conversation(item, i))
    return out
