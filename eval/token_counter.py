"""Exact token counts for the memory context, via Anthropic ``count_tokens``.

WHY NOT AN ESTIMATE
The published "memory tokens per question" figure was derived as
``total_input_tokens / n - 87`` (or ``- 110``): the mean answer-prompt input
tokens reported by the API, minus a once-measured constant standing in for the
prompt template plus the question. That constant is only right for the AVERAGE
question -- a long temporal question and a two-word single-hop question do not
carry the same overhead -- and it silently absorbs any prompt-template edit. It
is the one number in the Mem0 Table 2 comparison we most need to be able to
defend, so it is now MEASURED per question instead.

WHAT IS COUNTED
``count(text)`` returns ``client.messages.count_tokens(model, messages=[{"role":
"user", "content": text}]).input_tokens`` -- the count the answer model itself
would charge for that text, with the same tokenizer as the run. That number
includes the message ENVELOPE (the few tokens the API adds for the role framing
of a one-message request). ``envelope_tokens()`` measures the envelope once, so
a run can report both the raw count and the envelope-free count and say which is
which:

    T(S)  = envelope + t(S)
    T(SS) = envelope + t(SS)  and  t(SS) = 2*t(S) for a probe S chosen to
                                   concatenate on a token boundary
    => envelope = 2*T(S) - T(SS)

The probe is a long, newline-terminated, ASCII string, so the seam between the
two copies falls on a newline and cannot merge tokens.

COST AND PLACEMENT
``count_tokens`` runs no inference and is not billed. It is still a network
call, so every result is cached by SHA-256 of the text (process-wide, lock-
guarded) and every call site keeps it OUTSIDE the timed regions -- a token count
must never appear in a latency number.

DEGRADATION
No API key, no ``anthropic`` package, or a failing endpoint -> ``count`` returns
``None`` and warns ONCE. A missing count is reported as ``None``, never as a
guess, so a downstream mean is computed over the questions that actually have
one and the results file says how many that was.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# A long ASCII probe that ends on a newline, so probe+probe cannot merge tokens
# across the seam. 40 lines is comfortably above any model's minimum.
_PROBE = ("the quick brown fox jumps over the lazy dog while a memory engine "
          "fuses three ranked lists\n") * 40


class TokenCounter:
    """Cached ``count_tokens`` for one model. Thread-safe; never raises."""

    def __init__(self, model: str):
        self.model = model
        # OpenAI answer models (Mem0 protocol: gpt-4o-mini) are counted with
        # tiktoken -- the tokenizer that model is billed by -- because the
        # Anthropic count_tokens endpoint rejects non-Claude model ids.
        _m = (model or "").strip().lower()
        self._is_oai = _m.startswith(("gpt", "openai:", "o1", "o3", "o4"))
        self._enc = None
        self._cache: dict[str, int] = {}
        self._lock = threading.Lock()
        self._client = None
        self._broken = False
        self.calls = 0
        self.hits = 0

    def _get_client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415 — lazy: keeps anthropic optional

            self._client = anthropic.Anthropic()
        return self._client

    def _get_enc(self):
        if self._enc is None:
            import tiktoken  # noqa: PLC0415 - lazy

            name = self.model.split(":", 1)[1] if ":" in self.model else self.model
            try:
                self._enc = tiktoken.encoding_for_model(name)
            except KeyError:
                self._enc = tiktoken.get_encoding("o200k_base")
        return self._enc

    def count(self, text: str) -> int | None:
        if self._broken:
            return None
        if not text:
            text = ""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
        try:
            if self._is_oai:
                n = len(self._get_enc().encode(text))
            else:
                resp = self._get_client().messages.count_tokens(
                    model=self.model,
                    messages=[{"role": "user", "content": text or " "}],
                )
                n = int(resp.input_tokens)
        except Exception as exc:  # noqa: BLE001 — a token count must never fail a run
            if not self._broken:
                self._broken = True
                logger.warning(
                    "count_tokens unavailable (%s: %s) — context_tokens will be "
                    "None for this run and memory_tokens will fall back to the "
                    "input-token estimate. Fix the API key/network to get the "
                    "measured number.", type(exc).__name__, exc)
            return None
        with self._lock:
            self._cache[key] = n
            self.calls += 1
        return n

    def envelope_tokens(self) -> int | None:
        """Tokens the one-message envelope adds, measured (see module docstring)."""
        a = self.count(_PROBE)
        b = self.count(_PROBE + _PROBE)
        if a is None or b is None:
            return None
        return 2 * a - b

    def stats(self) -> dict:
        return {"model": self.model, "tokenizer": ("tiktoken" if self._is_oai else "anthropic.count_tokens"),
                "api_calls": self.calls,
                "cache_hits": self.hits, "available": not self._broken}


def make_token_counter(model: str) -> Callable[[str], int | None]:
    """``text -> exact token count`` for ``model`` (or ``None`` if unavailable)."""
    return TokenCounter(model).count
