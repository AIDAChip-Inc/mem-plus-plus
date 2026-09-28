"""Embeddings — protocol + preset config. Ported verbatim from production.

Nothing here imports heavy dependencies (numpy/onnxruntime), so this module is
always safe to import in test or offline environments.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

EmbeddingIntent = Literal["document", "query"]


def _is_blank(text: str | None) -> bool:
    """No non-whitespace content: ``None``, ``""``, or all whitespace."""
    return not text or text.isspace()


def first_nonblank(summary: str | None, content: str | None) -> str | None:
    """Prefer ``summary``, else ``content`` — a blank summary does NOT shadow it."""
    return content if _is_blank(summary) else summary


class EmbeddingPresetConfig(BaseModel):
    """Static configuration for a named embedding preset.

    This replica pins the ONNX backend (MiniLM-384); the torch/sentence-transformers
    backend is not carried over.
    """

    model: str
    dimensions: int
    document_prefix: str | None = None
    query_prefix: str | None = None
    # "openai" added so mem++ can be evaluated on the same embedding stack as
    # the systems it is compared against (mem0 uses text-embedding-3-small at
    # 1536-d). "onnx" remains the default and the published configuration.
    backend: Literal["onnx", "openai"] = "onnx"

    def build_payload(self, text: str, intent: EmbeddingIntent) -> str | None:
        """Blank text -> ``None``; else apply the intent's task prefix (if any)."""
        if _is_blank(text):
            return None
        prefix = self.query_prefix if intent == "query" else self.document_prefix
        return f"{prefix}: {text}" if prefix else text


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural interface for embedding backends: ``dimensions``, ``embed``, ``warm_load``."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str, *, intent: EmbeddingIntent = "document") -> list[float] | None:
        """Normalized embedding, or ``None`` when unavailable/blank."""
        ...

    def warm_load(self) -> bool:
        """Preload the model. Returns ``True`` iff ready."""
        ...
