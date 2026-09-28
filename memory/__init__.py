"""memory-research engine — a self-contained, faithful replica of the AIDAChip
team-memory system (recall + write core, ONNX MiniLM-384 embeddings, pgvector RRF).

Public API::

    from memory import recall_facts, store_facts, store_facts_verbatim
"""
from memory.recall import recall_facts, store_facts, store_facts_verbatim

__all__ = ["recall_facts", "store_facts", "store_facts_verbatim"]
