"""Embedding service factory — singleton registry keyed by preset name.

Consumers call ``get_embedding_service("memory")`` (same signature as production)
and receive the ONNX MiniLM-384 provider. This replica pins the ONNX backend;
the torch/sentence-transformers dispatch is not carried over.
"""
from __future__ import annotations

import threading

from memory import config
from memory.embeddings.provider import EmbeddingPresetConfig, EmbeddingProvider

_registry: dict[str, EmbeddingProvider] = {}
_lock = threading.Lock()

# The "memory" preset is read from config, so MEMORY_EMBEDDING_BACKEND=openai
# swaps MiniLM-384 for text-embedding-3-small at 1536-d without touching any
# call site. Default is unchanged: ONNX MiniLM-384, the published configuration.
_PRESETS: dict[str, dict] = {
    "memory": {
        "model": config.EMBEDDING_MODEL,
        "dimensions": config.EMBEDDING_DIM,
        "backend": config.MEMORY_EMBEDDING_BACKEND,
    },
}


def _build(cfg: EmbeddingPresetConfig) -> EmbeddingProvider:
    if cfg.backend == "openai":
        from memory.embeddings.openai_service import OpenAIEmbeddingService

        return OpenAIEmbeddingService(cfg)
    from memory.embeddings.onnx_service import OnnxEmbeddingService

    return OnnxEmbeddingService(cfg)


def get_embedding_service(preset: str) -> EmbeddingProvider:
    """Return a singleton EmbeddingProvider for the named preset (only ``"memory"``)."""
    if preset in _registry:
        return _registry[preset]
    with _lock:
        if preset not in _registry:
            _registry[preset] = _build(EmbeddingPresetConfig(**_PRESETS[preset]))
    return _registry[preset]


def reset_registry() -> None:
    """Clear all cached instances. **Test helper only.**"""
    with _lock:
        _registry.clear()
