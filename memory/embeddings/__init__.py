"""Unified embedding service — Protocol-based, factory-managed.

Public API::

    from memory.embeddings import get_embedding_service
    provider = get_embedding_service("memory")
    vector = provider.embed("some text")
    vector = provider.embed("a query", intent="query")
"""
from memory.embeddings.factory import get_embedding_service, reset_registry
from memory.embeddings.provider import (
    EmbeddingIntent,
    EmbeddingPresetConfig,
    EmbeddingProvider,
    first_nonblank,
)

# Shared loud-fallback diagnostic (design §8), reused by store.py.
EMBEDDER_UNAVAILABLE_HINT = (
    "in-process ONNX embedder could not load the fp32 all-MiniLM-L6-v2 model "
    "(missing/corrupt model file, or onnxruntime failed to init). Fix: confirm "
    "onnxruntime + tokenizers are installed and the model is fetchable from "
    "HuggingFace."
)

__all__ = [
    "EMBEDDER_UNAVAILABLE_HINT",
    "EmbeddingIntent",
    "EmbeddingPresetConfig",
    "EmbeddingProvider",
    "first_nonblank",
    "get_embedding_service",
    "reset_registry",
]
