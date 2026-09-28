"""Evaluation metrics for the memory-research harness.

Three families, all pure functions (no I/O, no LLM calls) so they are unit
tested with known inputs → known outputs:

  * Retrieval:  recall@k, MRR, nDCG@k        (ranked ids vs. a gold id set)
  * Answer:     token-F1, exact-match         (SQuAD-style normalization)
  * Systems:    latency, tokens, storage       (accounting helpers + aggregation)

The LLM-judge answer metric is NOT here — it needs a pinned model + published
prompt and lives in ``judge.py`` (kept separate so this module stays pure).
"""
from __future__ import annotations

import math
import re
import statistics
import string
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Answer-text normalization (SQuAD convention: lowercase, strip punctuation,
# drop the articles a/an/the, collapse whitespace).
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """SQuAD-style normalization used by both token_f1 and exact_match."""
    text = (text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between a prediction and a gold answer (SQuAD F1).

    Both-empty → 1.0; exactly-one-empty → 0.0; no shared tokens → 0.0.
    """
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    """BLEU-1: unigram precision against the gold, with the brevity penalty.

    Reported as B_1 in the mem0 comparison tables. Unigram-only (n=1) because a
    memory answer is typically a short factual span where higher-order n-gram
    overlap is dominated by phrasing rather than correctness. Clipped counts
    follow Papineni et al. (2002): a prediction token can match a gold token at
    most as many times as it occurs in the gold, so repeating a correct word
    cannot inflate the score.

    The brevity penalty is what stops a one-word answer that happens to hit a
    gold token from scoring 1.0 -- without it BLEU-1 rewards terseness, which on
    this benchmark would flatter the extraction arms.
    """
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    clipped = sum(min(c, gold_counts[t]) for t, c in Counter(pred_tokens).items())
    precision = clipped / len(pred_tokens)
    if precision == 0.0:
        return 0.0
    # brevity penalty: 1 when the prediction is at least as long as the gold
    bp = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(
        1.0 - len(gold_tokens) / len(pred_tokens))
    return bp * precision


def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 95)) -> dict:
    """p50 / p95 by linear interpolation -- the shape the mem0 latency table uses.

    A mean latency hides the tail, and the tail is what a user feels. Reported
    alongside the mean, never instead of it.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {f"p{int(p)}": None for p in ps}
    out = {}
    for p in ps:
        if len(vals) == 1:
            out[f"p{int(p)}"] = vals[0]
            continue
        idx = (p / 100.0) * (len(vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
        out[f"p{int(p)}"] = vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)
    return out


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalized prediction equals the normalized gold, else 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def best_over_golds(metric, prediction: str, golds: Sequence[str]) -> float:
    """Max of ``metric(prediction, g)`` over a list of acceptable gold answers.

    LongBench and LongMemEval both allow multiple reference answers; the score
    is the best match, per the SQuAD/LongBench convention.
    """
    if not golds:
        return 0.0
    return max(metric(prediction, g) for g in golds)


# ---------------------------------------------------------------------------
# Retrieval metrics. `retrieved_ids` is a RANK-ORDERED list (best first);
# `gold_ids` is an unordered set of relevant ids. `k` truncates the ranking.
# ---------------------------------------------------------------------------


def recall_at_k(gold_ids: Iterable[str], retrieved_ids: Sequence[str], k: int | None = None) -> float | None:
    """|gold ∩ top-k retrieved| / |gold|. Returns None when there is no gold
    evidence (nothing to score against — excluded from aggregates)."""
    gold = set(gold_ids)
    if not gold:
        return None
    top = retrieved_ids if k is None else retrieved_ids[:k]
    return len(gold & set(top)) / len(gold)


def mrr(gold_ids: Iterable[str], retrieved_ids: Sequence[str]) -> float:
    """Reciprocal rank of the FIRST relevant hit (1-indexed); 0.0 if none."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_ids: Iterable[str], retrieved_ids: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k. DCG uses 1/log2(rank+1) gain; IDCG is the ideal
    ranking (all relevant items first). 0.0 when there is no gold."""
    gold = set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
        if rid in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Multi-run variance — a single number is a noise candidate. Report mean ±1σ.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStats:
    mean: float
    stddev: float
    n_runs: int


def aggregate_runs(values: Sequence[float]) -> RunStats:
    """mean / population-σ / n over per-run scalars. σ=0.0 for a single run."""
    vals = [v for v in values if v is not None]
    if not vals:
        return RunStats(mean=0.0, stddev=0.0, n_runs=0)
    mean = statistics.fmean(vals)
    stddev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return RunStats(mean=mean, stddev=stddev, n_runs=len(vals))


# ---------------------------------------------------------------------------
# Systems metrics: latency, token cost, storage footprint.
# ---------------------------------------------------------------------------


@contextmanager
def measure_latency():
    """Context manager yielding a 1-element list; on exit element 0 holds the
    wall-clock seconds. Usage: ``with measure_latency() as t: ...`` then
    ``t[0]``."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


# Per-1M-token prices for the pinned judge/answer model (Haiku 4.5), used only
# for a rough $ estimate in the summary; not a billing source of truth.
INPUT_PRICE_PER_MTOK = 1.00
OUTPUT_PRICE_PER_MTOK = 5.00


def token_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost estimate at the pinned model's list price."""
    return (input_tokens * INPUT_PRICE_PER_MTOK + output_tokens * OUTPUT_PRICE_PER_MTOK) / 1e6


def storage_footprint_bytes(facts: Iterable[str]) -> int:
    """UTF-8 byte footprint of the stored fact strings — a proxy for the memory
    system's storage cost on a corpus (accuracy that costs 10× storage is
    reported as exactly that)."""
    return sum(len(f.encode("utf-8")) for f in facts)
