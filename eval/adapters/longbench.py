"""LongBench (v1) adapter: per-config JSONL → common format.

Source: HF ``THUDM/LongBench`` (fetched by ``datasets/fetch.py longbench`` into
``datasets/longbench/<config>.jsonl``). Each row is one long-context task
instance: ``input`` (question), ``context`` (the long document), ``answers``
(acceptable references), ``all_classes``, ``length``, ``dataset``.

LongBench measures answer quality against a single long document — there is no
turn-level evidence annotation, so we model each row as a Conversation whose
turns are context chunks (ingestable/recallable) and whose single QAItem has no
``gold_turn_ids`` (retrieval metrics are skipped for it; answer metrics apply).
The config/task name is the category. Per-task LongBench metrics (F1 / ROUGE-L /
acc / edit-sim / EM) are computed downstream; token-F1 + LLM-judge give a
protocol-consistent cross-benchmark number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import Conversation, QAItem, Turn

MAX_CHUNKS = 200
CHUNK_WORDS = 200


def _chunk(context: str) -> list[str]:
    """Split the long document into ingestable chunks: paragraphs first, then
    fixed word windows for any oversized paragraph. Capped at MAX_CHUNKS."""
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", context or ""):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= CHUNK_WORDS:
            chunks.append(para)
        else:
            for i in range(0, len(words), CHUNK_WORDS):
                chunks.append(" ".join(words[i:i + CHUNK_WORDS]))
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks[:MAX_CHUNKS]


def _to_conversation(row: dict, category: str, idx: int) -> Conversation:
    turns = [
        Turn(turn_id=f"{category}:{idx}:c{c}", speaker="document", text=chunk)
        for c, chunk in enumerate(_chunk(row.get("context", "")))
    ]
    answers = row.get("answers") or ([] if row.get("answer") is None else [row["answer"]])
    qa = [QAItem(
        qa_id=f"{category}:{idx}",
        question=str(row.get("input", row.get("question", ""))),
        gold_answers=[str(a) for a in answers],
        category=category,
        gold_turn_ids=set(),  # LongBench has no turn-level evidence annotation
        abstention=False,
    )]
    return Conversation(conversation_id=f"longbench_{category}_{idx}", turns=turns, qa=qa)


def load(path: str | Path) -> list[Conversation]:
    """Load LongBench rows. ``path`` may be a single ``<config>.jsonl`` or a
    directory of them (the fetch layout); the config stem becomes the category."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    conversations: list[Conversation] = []
    for f in files:
        category = f.stem
        for idx, line in enumerate(f.read_text().splitlines()):
            line = line.strip()
            if line:
                conversations.append(_to_conversation(json.loads(line), category, idx))
    return conversations
