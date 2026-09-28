"""OpenAI embedding backend — text-embedding-3-small / -large.

Added so mem++ can be evaluated on the same embedding stack as the systems it is
compared against. The published mem++ arms use the local ONNX MiniLM-384; mem0's
own pipeline uses text-embedding-3-small at 1536 dimensions, a 4x wider vector.
Holding the embedder constant removes the last uncontrolled variable between the
two systems on LoCoMo.

Implements the same structural interface as OnnxEmbeddingService
(memory/embeddings/provider.py EmbeddingProvider): ``dimensions``, ``embed``,
``warm_load``. Returns L2-normalised vectors, matching the ONNX service, so the
pgvector cosine operator (``<=>``) keeps its meaning and no ranking code changes.

NOTE: the embedding column is fixed-width (``Vector(384)`` in models.py and in
alembic 0001). A 1536-d run therefore needs its OWN database -- see
scripts/make_1536_db.py. Do not point this provider at the 384-d corpus.
"""
from __future__ import annotations

import logging
import math
import time
import os
import threading

from memory.embeddings.provider import EmbeddingIntent, EmbeddingPresetConfig


logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """Embeddings via the OpenAI API. One client per process, lazily built."""

    def __init__(self, cfg: EmbeddingPresetConfig) -> None:
        self._cfg = cfg
        self._client = None
        self._lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._cfg.dimensions

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI

                    key = os.environ.get("OPENAI_API_KEY")
                    if not key:
                        raise RuntimeError(
                            "OPENAI_API_KEY is not set; the OpenAI embedding backend "
                            "cannot run. Set it or switch MEMORY_EMBEDDING_BACKEND=onnx."
                        )
                    self._client = OpenAI(api_key=key)
        return self._client


    # ---- model input cap -------------------------------------------------
    # text-embedding-3-* reject ANY single input over 8,192 tokens with a 400,
    # and a 400 kills the whole batch, not just the offending row. LoCoMo's
    # longest turn is 146 tokens and OrgMemBench's artefacts are well inside the
    # cap, but LongMemEval-S carries a few very long turns: the full 500-instance
    # run died at instance 31 on `Invalid 'input[240]': maximum input length is
    # 8192 tokens`.
    #
    # The remedy is to embed the first 8,191 tokens of such a row. This is a
    # DISCLOSED approximation, not a silent one:
    #   * the row's TEXT is stored and delivered to the answerer in full -- only
    #     the vector is computed from the prefix, so the lexical leg and the
    #     answer prompt still see every token;
    #   * ``clamped()`` reports how many rows it affected, so a run can state the
    #     exact count instead of asserting the cap never binds.
    _MAX_INPUT_TOKENS = 8191
    _clamped = 0
    _encoding = None

    @classmethod
    def clamped(cls) -> int:
        """Rows whose embedding was computed from a truncated prefix."""
        return cls._clamped

    def _clamp(self, payload: str) -> str:
        # Cheap reject: a token is >= 1 character, so anything under the cap in
        # characters is under it in tokens and never needs encoding.
        if len(payload) <= self._MAX_INPUT_TOKENS:
            return payload
        cls = type(self)
        if cls._encoding is None:
            import tiktoken  # noqa: PLC0415 -- lazy; only long inputs pay for it

            try:
                cls._encoding = tiktoken.encoding_for_model(self._cfg.model)
            except KeyError:
                cls._encoding = tiktoken.get_encoding("cl100k_base")
        toks = cls._encoding.encode(payload)
        if len(toks) <= self._MAX_INPUT_TOKENS:
            return payload
        cls._clamped += 1
        out = cls._encoding.decode(toks[: self._MAX_INPUT_TOKENS])
        logger.warning(
            "embedding input of %d tokens exceeds the %d-token model cap; "
            "embedding its first %d tokens (row %d clamped this run). The stored "
            "text is unchanged.", len(toks), self._MAX_INPUT_TOKENS,
            self._MAX_INPUT_TOKENS, cls._clamped)
        return out

    def embed(self, text: str, *, intent: EmbeddingIntent = "document") -> list[float] | None:
        """Normalized embedding, or ``None`` for blank text.

        ``intent`` is accepted for interface parity; text-embedding-3-* is a
        symmetric model with no task prefixes, so document and query embeddings
        are produced identically (the same is true of the MiniLM service).
        """
        payload = self._cfg.build_payload(text, intent)
        if payload is None:
            return None
        payload = self._clamp(payload)
        resp = self._get_client().embeddings.create(
            model=self._cfg.model, input=payload, dimensions=self._cfg.dimensions
        )
        vec = list(resp.data[0].embedding)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            return None
        return [v / norm for v in vec]

    # Inputs per request. The API's hard cap is 2048, but the binding limit in
    # practice is tokens-per-minute, not inputs-per-call: one request carrying a
    # whole corpus (5,882 LoCoMo turns) exhausts the quota outright, and
    # concurrent ingests multiply it. Measured: three parallel ingests returned
    # RateLimitError on every attempt and lost two cells after 10 retries each.
    # Chunking bounds the per-request cost so the backoff below can actually
    # drain the bucket instead of hammering a closed door.
    _BATCH = 256
    _RATE_LIMIT_BACKOFF = (5.0, 15.0, 40.0, 90.0, 150.0, 150.0)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Batched embed, chunked and rate-limit aware.

        A rate limit is a request to WAIT, not a transient error to retry
        quickly. The backoff here is deliberately long and lives below the
        driver-level retry wrapper, which cannot tell the two apart.
        """
        idx = [i for i, t in enumerate(texts) if self._cfg.build_payload(t, "document")]
        out: list[list[float] | None] = [None] * len(texts)
        if not idx:
            return out
        client = self._get_client()
        for start in range(0, len(idx), self._BATCH):
            group = idx[start:start + self._BATCH]
            payloads = [self._clamp(self._cfg.build_payload(texts[i], "document"))
                        for i in group]
            resp = None
            last = len(self._RATE_LIMIT_BACKOFF) - 1
            for attempt, wait in enumerate(self._RATE_LIMIT_BACKOFF):
                try:
                    resp = client.embeddings.create(
                        model=self._cfg.model, input=payloads,
                        dimensions=self._cfg.dimensions)
                    break
                except Exception as exc:
                    if type(exc).__name__ != "RateLimitError" or attempt == last:
                        raise
                    logger.warning(
                        "embedding rate limit; waiting %.0fs (rows %d-%d of %d)",
                        wait, start, start + len(group), len(idx))
                    time.sleep(wait)
            for slot, item in zip(group, resp.data, strict=True):
                vec = list(item.embedding)
                norm = math.sqrt(sum(v * v for v in vec))
                out[slot] = [v / norm for v in vec] if norm > 0 else None
        return out

    def warm_load(self) -> bool:
        """No model to preload; verify credentials and dimensionality instead."""
        try:
            v = self.embed("warm load probe")
            return v is not None and len(v) == self._cfg.dimensions
        except Exception:
            return False
