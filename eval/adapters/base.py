"""Common conversation + QA format that every benchmark adapter emits.

Each adapter maps a benchmark's native JSON into a list of ``Conversation``s:
turns to ingest into memory, plus QA items to ask. Retrieval scoring works off
``Turn.turn_id`` (the provenance id) vs. ``QAItem.gold_turn_ids``; answer
scoring works off ``QAItem.gold_answers``; abstention items reward not-knowing.

No-leakage is structural: ingestion consumes only ``turns``, never ``qa``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    """One ingestable memory unit (a conversation turn)."""

    turn_id: str
    speaker: str
    text: str
    occurred_at: datetime | None = None

    def render(self) -> str:
        """Verbatim fact text stored in memory: ``[occurred_at] speaker: text``."""
        stamp = self.occurred_at.isoformat() if self.occurred_at else "?"
        return f"[{stamp}] {self.speaker}: {self.text}"


@dataclass
class QAItem:
    """One question with its gold answer(s), category, and gold evidence."""

    qa_id: str
    question: str
    gold_answers: list[str]
    category: str
    gold_turn_ids: set[str] = field(default_factory=set)
    abstention: bool = False
    # "Today" for the question (LongMemEval ``question_date``). When set, the
    # harness bounds recall to rows with ``occurred_at <= question_date`` and
    # gives the answerer this date as the current date. ``None`` = undated.
    question_date: datetime | None = None

    @property
    def primary_gold(self) -> str:
        return self.gold_answers[0] if self.gold_answers else ""


@dataclass
class Conversation:
    """A conversation: turns to ingest + questions to ask."""

    conversation_id: str
    turns: list[Turn]
    qa: list[QAItem]
