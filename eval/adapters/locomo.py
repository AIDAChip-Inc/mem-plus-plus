"""LoCoMo adapter: locomo10.json → common Conversation/QAItem format.

Source: snap-research/locomo, ``data/locomo10.json`` (fetched by
``datasets/fetch.py locomo``). Each sample carries a ``conversation`` dict with
``session_N`` turn lists (+ ``session_N_date_time``) and a ``qa`` list.

Turn provenance = the dataset's ``dia_id`` (gold evidence ids reference it).
Categories (per the paper / repo eval code): 1=multi_hop, 2=temporal,
3=open_domain, 4=single_hop, 5=adversarial. Adversarial items are unanswerable
(carry ``adversarial_answer``, no ``answer``) — abstention is the correct
behavior, so the gold answer is the abstention sentinel.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .base import Conversation, QAItem, Turn

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
ADVERSARIAL_GOLD = "This question is not answerable from the conversation."

_SESSION_KEY = re.compile(r"^session_(\d+)$")
_DT_FORMATS = (
    "%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y", "%I:%M %p on %B %d, %Y",
    "%H:%M on %d %B, %Y", "%d %B, %Y", "%d %B %Y", "%B %d, %Y",
)


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


def _photo_tag(cell: dict) -> str:
    """Byte-identical to mem0's shipped photo handling (locomo/run.py
    session_to_chunks, mirrored in drivers/mem0_adapter.py:156) so both systems
    ingest the same characters."""
    blip, query = cell.get("blip_caption", ""), cell.get("query", "")
    if query and blip:
        return f"[Sharing image - query: {query}. The image shows: {blip}]"
    if query:
        return f"[Sharing image - query for: {query}]"
    if blip:
        return f"[Sharing image that shows: {blip}]"
    return ""


def _evidence_ids(evidence) -> set[str]:
    """Normalize gold evidence; the dataset has joined entries like 'D8:6; D9:17'."""
    ids: set[str] = set()
    for entry in evidence or []:
        for part in re.split(r"[;,]", str(entry)):
            if part.strip():
                ids.add(part.strip())
    return ids


def _to_conversation(sample: dict, idx: int) -> Conversation:
    conv = sample["conversation"]
    turns: list[Turn] = []
    for key in sorted(conv, key=lambda k: int(m.group(1)) if (m := _SESSION_KEY.match(k)) else 1 << 30):
        m = _SESSION_KEY.match(key)
        if not m:
            continue
        occurred = _parse_dt(conv.get(f"{key}_date_time"))
        for cell in conv[key]:
            dia_id = cell.get("dia_id") or f"{key}:{len(turns)}"
            text = str(cell.get("text", "")).strip()
            # LoCoMo turns can carry an image with a BLIP caption and a search
            # query. mem0's shipped pipeline ingests both (locomo/run.py
            # session_to_chunks); this adapter used to drop them, so mem0 was
            # scored on 68,046 characters of evidence mem++ never saw --
            # 1,226 of 5,882 turns (20.8%), and 700 of the 1,540 four-category
            # questions have at least one gold turn carrying image content. On
            # the 84 questions whose answer appears ONLY in the caption, mem++
            # scored 14.3 against mem0's 56.0. Ingesting it makes the corpus
            # identical for every system.
            tag = _photo_tag(cell)
            if tag:
                text = f"{text} {tag}" if text else tag
            turns.append(Turn(
                turn_id=str(dia_id),
                speaker=str(cell.get("speaker", "?")),
                text=text,
                occurred_at=occurred,
            ))

    qa: list[QAItem] = []
    for j, item in enumerate(sample.get("qa", [])):
        cat_raw = item.get("category")
        category = CATEGORY_NAMES.get(cat_raw, str(cat_raw))
        abstention = category == "adversarial" or "adversarial_answer" in item
        gold = ADVERSARIAL_GOLD if abstention else str(item.get("answer", ""))
        qa.append(QAItem(
            qa_id=f"conv{idx}:q{j}",
            question=str(item.get("question", "")),
            gold_answers=[gold],
            category=category,
            gold_turn_ids=_evidence_ids(item.get("evidence")),
            abstention=abstention,
        ))
    return Conversation(conversation_id=f"locomo_{idx}", turns=turns, qa=qa)


def load(path: str | Path) -> list[Conversation]:
    """Load locomo10.json into the common format."""
    data = json.loads(Path(path).read_text())
    return [_to_conversation(sample, i) for i, sample in enumerate(data)]
