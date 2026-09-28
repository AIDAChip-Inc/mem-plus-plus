"""In-process ONNX embedding service. Ported verbatim from production.

Runs all-MiniLM-L6-v2 (384-d) via ``onnxruntime`` + a HF ``tokenizers`` tokenizer
+ numpy mean-pool & L2-normalize — no torch, cross-platform. Lazy-loads on first
``embed()``. On any load failure the instance latches ``_load_failed`` and every
subsequent ``embed()`` returns ``None`` (never raises into consumers). The fp32
``onnx/model.onnx`` export is pinned by commit so vectors share one cosine space.
"""
from __future__ import annotations

import logging

import numpy as np

from memory.embeddings.provider import EmbeddingIntent, EmbeddingPresetConfig

logger = logging.getLogger(__name__)

# fp32 export of all-MiniLM-L6-v2, pinned by commit for exact/auditable loading.
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_MAX_SEQ_LENGTH = 256  # all-MiniLM-L6-v2 max_seq_length


def _resolve_repo(model: str) -> str:
    """Bare model name -> canonical sentence-transformers HF repo id."""
    return model if "/" in model else f"sentence-transformers/{model}"


def mean_pool_l2_normalize(
    last_hidden_state: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Attention-masked mean pool + L2 normalize — matches sentence-transformers."""
    mask = attention_mask.astype(np.float32)[..., np.newaxis]
    summed = (last_hidden_state.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    mean = summed / counts
    norm = np.clip(np.linalg.norm(mean, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return mean / norm


class OnnxEmbeddingService:
    """In-process onnxruntime embedding provider (EmbeddingProvider protocol)."""

    def __init__(self, config: EmbeddingPresetConfig) -> None:
        self._config = config
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._load_failed = False

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    def embed(self, text: str, *, intent: EmbeddingIntent = "document") -> list[float] | None:
        payload = self._config.build_payload(text, intent)
        if payload is None:
            return None
        self._load()
        if self._session is None:  # tokenizer is set in lockstep with the session
            return None
        try:
            return self._encode(payload)
        except Exception as exc:
            logger.warning("ONNX embedding inference failed: %s", exc)
            return None

    # Rows per onnxruntime call. The single-text ``embed`` path pays the full
    # graph-launch + BLAS-setup overhead once PER ROW; batching amortises it over
    # ``_BATCH`` rows. 64 keeps the padded (64, 256, 384) activation well inside
    # cache-friendly territory on a laptop CPU.
    _BATCH = 64

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Batched ``embed`` — one onnxruntime call per ``_BATCH`` texts.

        Semantically identical to ``[self.embed(t) for t in texts]``: the same
        payload build, the same truncation, the same attention-masked mean-pool +
        L2 normalize. Padding is added only to square the batch and is zeroed in
        ``attention_mask``, so BERT's extended attention mask excludes it and the
        real tokens see exactly the context they see alone (verified against the
        single-text path by cosine, see the LongMemEval ingest parity artefact).
        """
        out: list[list[float] | None] = [None] * len(texts)
        payloads = [(i, p) for i, t in enumerate(texts)
                    if (p := self._config.build_payload(t, "document")) is not None]
        if not payloads:
            return out
        self._load()
        if self._session is None:
            return out
        # Length-bucket so a batch is padded to ITS OWN longest member rather than
        # to the longest text anywhere in the call. LongMemEval turns range from
        # ~5 to 256 tokens; unsorted, nearly every batch pads to the 256 cap and
        # most of the matmul is spent on masked-out positions. Sorting changes no
        # output (padding is excluded by the attention mask either way) because
        # each result is written back to its own original slot.
        payloads.sort(key=lambda p: len(p[1]))
        try:
            for start in range(0, len(payloads), self._BATCH):
                group = payloads[start:start + self._BATCH]
                encs = self._tokenizer.encode_batch([p for _i, p in group])
                width = max(len(e.ids) for e in encs)
                ids = np.zeros((len(encs), width), dtype=np.int64)
                mask = np.zeros((len(encs), width), dtype=np.int64)
                types = np.zeros((len(encs), width), dtype=np.int64)
                for r, e in enumerate(encs):
                    n = len(e.ids)
                    ids[r, :n] = e.ids
                    mask[r, :n] = e.attention_mask
                    types[r, :n] = e.type_ids
                feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
                outputs = self._session.run(
                    None, {k: v for k, v in feeds.items() if k in self._input_names}
                )
                hidden = next((a for o in outputs if (a := np.asarray(o)).ndim == 3), None)
                if hidden is None:
                    return out
                pooled = mean_pool_l2_normalize(hidden, mask).astype(np.float32)
                for (slot, _p), vec in zip(group, pooled, strict=True):
                    out[slot] = vec.tolist()
        except Exception as exc:
            logger.warning("ONNX batched embedding inference failed: %s", exc)
        return out

    def warm_load(self) -> bool:
        self._load()
        return self._session is not None

    def _load(self) -> None:
        if self._session is not None or self._load_failed:
            return
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            repo = _resolve_repo(self._config.model)
            model_path = hf_hub_download(repo, _ONNX_FILE, revision=_MODEL_REVISION)
            tokenizer_path = hf_hub_download(repo, _TOKENIZER_FILE, revision=_MODEL_REVISION)
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=_MAX_SEQ_LENGTH)
            self._tokenizer = tokenizer
            self._session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_names = {i.name for i in self._session.get_inputs()}
            logger.info("Loaded ONNX embedding model: %s (%s)", repo, _ONNX_FILE)
        except Exception as exc:
            logger.warning("Failed to load ONNX embedding model: %s", exc)
            self._session = None
            self._tokenizer = None
            self._load_failed = True

    def _encode(self, text: str) -> list[float] | None:
        enc = self._tokenizer.encode(text)
        attention_mask = np.asarray([enc.attention_mask], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.asarray([enc.ids], dtype=np.int64),
            "attention_mask": attention_mask,
            "token_type_ids": np.asarray([enc.type_ids], dtype=np.int64),
        }
        outputs = self._session.run(
            None, {k: v for k, v in feeds.items() if k in self._input_names}
        )
        # base-model export exposes the token-level (batch, seq, hidden) tensor.
        hidden = next((a for o in outputs if (a := np.asarray(o)).ndim == 3), None)
        if hidden is None:
            return None
        return mean_pool_l2_normalize(hidden, attention_mask)[0].astype(np.float32).tolist()
