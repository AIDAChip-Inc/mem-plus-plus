"""Evaluation harness: ingest → recall → answer → judge → score.

Given a list of ``Conversation``s (from any adapter) and a memory backend
(real engine wrapper or stub), for each conversation:

  1. INGEST every turn verbatim into a conversation-scoped persona (isolation
     between conversations; no-leakage — QA is never shown to ingestion).
  2. For each question: recall top-k, map recalled facts back to turn ids for
     retrieval scoring, build a context, generate an answer, judge it.
  3. Score retrieval (recall@k/MRR/nDCG@k, only where gold evidence exists),
     answer (token-F1/EM/LLM-judge), and systems (latency/tokens/storage).

Emits a results dict (config + per-QA rows + summary) and prints a table.
``recall_only=True`` skips the answer/judge LLM calls (retrieval sweep at $0).
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from . import metrics
from .adapters.base import Conversation, Turn
from .judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION, judge_accuracy, judge_one
from .llm import (
    ANSWER_MODEL,
    ANSWER_PROMPT_VERSION,
    LLMReply,
    build_answer_prompt,
    resolve_model,
)
from .memory_client import DEFAULT_K, HALF_LIFE_DAYS, RRF_K, RRF_WEIGHTS
from .sysinfo import describe_machine
from .token_counter import TokenCounter

AnswerFn = Callable[[str], LLMReply]
JudgeFn = Callable[[str], str]

# Fixed seed for deterministic fraction sub-sampling — a run at a given
# ``sample_fraction`` always selects the SAME QA items, so quick-iteration runs
# are reproducible and comparable across sessions. Changing this reshuffles the
# sample, so it is pinned like a protocol constant.
SAMPLE_SEED = 1234

# Upper bound on parallel workers. Caps concurrent answer+judge LLM calls
# (Anthropic rate limits) AND concurrent DB sessions (the engine's SQLAlchemy
# pool is ~15 connections: pool_size 5 + overflow 10). 8 stays comfortably
# under both; ``workers`` above this is clamped with a warning rather than
# risking 429s or pool exhaustion.
MAX_WORKERS = 8


@dataclass
class QAResult:
    qa_id: str
    category: str
    abstention: bool
    has_gold_evidence: bool
    recall_at_k: float | None
    mrr: float
    ndcg_at_k: float
    token_f1: float | None
    exact_match: float | None
    bleu1: float | None
    judged_correct: bool | None
    # SEARCH latency, mem0 Table 2 column 1: wall clock around the ONE
    # ``memory.recall_facts(...)`` call and nothing else — question text in,
    # delivered cards out. It therefore INCLUDES the query embedding, all four
    # SQL legs, RRF fusion, row hydration, and whatever a driver's recall
    # monkeypatch does inside it (graph expansion in con_graph3 / con_mem0g).
    # It EXCLUDES prompt assembly, the answer call, the judge, and token
    # counting.
    latency_s: float
    # TOTAL latency, mem0 Table 2 column 2: ``latency_s + answer_latency_s``.
    # The judge is a MEASUREMENT instrument, not part of the system under test,
    # so judge time is never in here.
    total_latency_s: float
    input_tokens: int
    output_tokens: int
    # Answer-generation wall clock alone, so total = latency_s + this exactly and
    # a reader can decompose the column instead of taking the sum on trust.
    answer_latency_s: float = 0.0
    # Answer-prompt input tokens as REPORTED BY THE API's usage block (prompt
    # template + question + memory excerpts + message envelope). Same value as
    # ``input_tokens``; named so the summary can distinguish "what the API
    # billed" from "what the memory block cost".
    prompt_input_tokens: int = 0
    # MEASURED tokens of the memory-excerpt block handed to the answerer — the
    # ``context`` string passed to build_answer_prompt — counted by the
    # Anthropic count_tokens endpoint for the answer model, OUTSIDE every timed
    # region. This replaces the ``total_input_tokens/n - 87`` estimate. None
    # when the counter was unavailable (never a guess).
    context_tokens: int | None = None
    # Ordered identity of the delivered cards: sha1(summary)[:12] per row, in
    # rank order. The engine returns no row id, and a driver's recall patch can
    # synthesise rows that have none, so the identity used is the CONTENT that
    # actually reached the answerer — which is exactly what a replay has to
    # reproduce for a re-measured latency to be a latency for THIS run.
    retrieved_ids: list[str] = field(default_factory=list)
    # Gold-scoring ids: recalled cards mapped back to turn ids (rank-ordered,
    # de-duplicated) — the list recall@k / MRR / nDCG@k are computed from.
    retrieved_turn_ids: list[str] = field(default_factory=list)
    # The generated answer, kept verbatim so a run can be RE-JUDGED without
    # being re-run. Without this, "judge with a second model" means paying for
    # the whole pipeline twice, and any judge fixed after the fact orphans every
    # number produced before it.
    answer_text: str | None = None
    # Question + gold travel with the answer so a results file is SELF-CONTAINED
    # for re-judging: rejudge.py can rebuild the judge prompt exactly, with no
    # access to the dataset and no re-running. (These are LoCoMo-derived
    # excerpts; see NOTICE -- they stay under CC BY-NC 4.0.)
    question: str | None = None
    primary_gold: str | None = None


def _map_recall_to_turn_ids(
    recalled: Sequence[dict], turns: Sequence[Turn], by_render: dict[str, str]
) -> list[str]:
    """Map recalled facts (dicts with a 'summary') back to turn ids, preserving
    rank order and de-duplicating. Exact render match first, then a containment
    fallback (the engine may return a consolidated summary, not the raw turn).

    ``by_render`` (render-text → turn_id) is passed in so it is built once per
    conversation, not rebuilt on every question."""
    out: list[str] = []
    seen: set[str] = set()
    for item in recalled:
        summary = (item.get("summary") or "").strip()
        turn_id = by_render.get(summary)
        if turn_id is None:
            for t in turns:
                if t.text and (t.text in summary or summary in t.render()):
                    turn_id = t.turn_id
                    break
        if turn_id and turn_id not in seen:
            seen.add(turn_id)
            out.append(turn_id)
    return out


def _build_context(recalled: Sequence[dict]) -> str:
    lines = [f"{i}. {(item.get('summary') or '').strip()}" for i, item in enumerate(recalled, 1)]
    return "\n".join(lines) if lines else "(no relevant memories found)"


def _card_ids(recalled: Sequence[dict]) -> list[str]:
    """Rank-ordered content identity of the delivered cards (see ``retrieved_ids``)."""
    return [hashlib.sha1((item.get("summary") or "").strip().encode("utf-8"),
                         usedforsecurity=False).hexdigest()[:12]
            for item in recalled]


def select_qa_fraction(
    conversations: Sequence[Conversation], fraction: float
) -> list[Conversation]:
    """Deterministically sub-sample to ``fraction`` of the benchmark's QA items.

    Selection method (documented for reproducibility): the QA items are flattened
    across conversations in their natural ``(conversation, question)`` order into a
    global index space of size N; ``ceil(fraction * N)`` indices are drawn WITHOUT
    replacement by ``random.Random(SAMPLE_SEED)`` — a fixed seed, so the same
    ``fraction`` always yields the same sample (spread across the dataset rather
    than head-sliced, so a small fraction stays representative). Each surviving
    conversation keeps its questions in natural order and its FULL turn list (all
    turns are needed to ingest before recall); conversations with zero selected
    questions are dropped so nothing is ingested for nothing.

    ``fraction == 1.0`` is a no-op pass-through. Raises ``ValueError`` outside
    ``(0, 1]``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")
    conversations = list(conversations)
    if fraction == 1.0:
        return conversations

    flat = [(ci, qi) for ci, conv in enumerate(conversations) for qi in range(len(conv.qa))]
    n = len(flat)
    if n == 0:
        return conversations
    n_sel = max(1, math.ceil(fraction * n))
    chosen = set(random.Random(SAMPLE_SEED).sample(flat, n_sel))

    out: list[Conversation] = []
    for ci, conv in enumerate(conversations):
        kept = [conv.qa[qi] for qi in range(len(conv.qa)) if (ci, qi) in chosen]
        if kept:
            out.append(Conversation(conversation_id=conv.conversation_id,
                                    turns=conv.turns, qa=kept))
    return out


def _run_conversation(
    conv: Conversation,
    memory: Any,
    *,
    answer_fn: AnswerFn | None,
    judge_fn: JudgeFn | None,
    k: int,
    recall_only: bool,
    persona: str,
    reuse_ingest: bool = False,
    update_hits: bool = False,
    on_question: Callable[[], None] | None = None,
    token_counter: TokenCounter | None = None,
) -> tuple[list[QAResult], int, bool]:
    """Ingest one conversation then score all its questions. Self-contained unit
    of work — the parallel granularity. Ingest completes before any recall, and
    each engine call (``count_facts`` / ``store_facts_verbatim`` / ``recall_facts``)
    opens and closes its OWN ``get_session()`` internally, so no SQLAlchemy Session
    is ever shared across threads; conversations use disjoint personas so their stub
    buckets never collide either.

    ``reuse_ingest`` skips the ingest step when this conversation's scope ALREADY
    holds memories (``memory.count_facts(conv_persona) > 0``) — recall then runs
    against the prior ingest, which is byte-identical to a fresh ingest of the same
    turns (the engine dedups by summary; the stub keeps one copy). If reuse is asked
    but the scope is empty, we ingest anyway so recall is never silently empty.
    Returns ``(rows, storage_bytes, ingested)`` — ``ingested`` is False only when the
    ingest was skipped via reuse. ``on_question`` (if given) fires once per completed
    question, for progress reporting. The storage footprint is a property of the
    turns, so it is reported identically whether or not this run re-ingested.

    ``update_hits`` (default False) controls the engine's recall hit/usage
    WRITEBACK. It must stay off for measurement: a conversation's questions all
    recall against ONE shared scope, and a writeback bumps ``hit_count`` on every
    row it returns. ``hit_count`` is the engine's SECOND ranking key (and worth
    ``10*log1p(hits)`` in salience), so with the writeback on, question N's ranking
    depends on questions 1..N-1 — retrieval scores then move with QA ORDER rather
    than with retrieval quality, and no A/B is attributable. Off, every question
    sees the identical corpus state and the run is order-independent. Set it True
    only to deliberately study the usage-feedback loop itself."""
    conv_persona = f"{persona}:{conv.conversation_id}"
    facts = [t.render() for t in conv.turns]
    by_render = {fact: turn.turn_id for fact, turn in zip(facts, conv.turns, strict=True)}
    ingested = not (reuse_ingest and memory.count_facts(conv_persona) > 0)
    if ingested:
        memory.store_facts_verbatim(conv_persona, facts)
    storage = metrics.storage_footprint_bytes(facts)

    rows: list[QAResult] = []
    for item in conv.qa:
        # ── SEARCH latency: this block and ONLY this block. ──────────────────
        # Nothing above it (context assembly, token counting) and nothing below
        # it (answering, judging) is inside the timer. A driver that monkeypatches
        # MemoryClient.recall_facts — con_matrix._recall, con_mem0g._recall,
        # con_graph3._recall — has its graph expansion timed here too, which is
        # correct: that work is part of producing the cards the answerer sees.
        with metrics.measure_latency() as t:
            recalled = memory.recall_facts(
                conv_persona, item.question, k=k, update_hits=update_hits
            )
        latency = t[0]
        # ── end of the timed region ─────────────────────────────────────────
        card_ids = _card_ids(recalled)
        context = _build_context(recalled)
        retrieved_ids = _map_recall_to_turn_ids(recalled, conv.turns, by_render)

        has_gold = bool(item.gold_turn_ids)
        r_at_k = metrics.recall_at_k(item.gold_turn_ids, retrieved_ids, k) if has_gold else None
        rr = metrics.mrr(item.gold_turn_ids, retrieved_ids) if has_gold else 0.0
        ndcg = metrics.ndcg_at_k(item.gold_turn_ids, retrieved_ids, k) if has_gold else 0.0

        tf1 = em = b1 = None
        judged: bool | None = None
        in_tok = out_tok = 0
        answer_text: str | None = None
        answer_latency = 0.0
        if not recall_only:
            prompt = build_answer_prompt(context=context, question=item.question)
            # ── TOTAL latency = search + THIS. The judge below is excluded. ──
            with metrics.measure_latency() as ta:
                reply = answer_fn(prompt)  # type: ignore[misc]
            answer_latency = ta[0]
            # ── end of the timed region ─────────────────────────────────────
            in_tok, out_tok = reply.input_tokens, reply.output_tokens
            answer_text = reply.text
            tf1 = metrics.best_over_golds(metrics.token_f1, reply.text, item.gold_answers)
            em = metrics.best_over_golds(metrics.exact_match, reply.text, item.gold_answers)
            b1 = metrics.best_over_golds(metrics.bleu1, reply.text, item.gold_answers)
            judged = judge_one(
                judge_fn,  # type: ignore[arg-type]
                question=item.question,
                gold_answer=item.primary_gold,
                generated_answer=reply.text,
                abstention=item.abstention,
            )

        # Exact memory-token count. Deliberately LAST: count_tokens is a network
        # call, and putting it here keeps it outside both timed regions by
        # construction rather than by comment. Runs on recall-only sweeps too —
        # it costs no inference, so a $0 retrieval sweep still reports MEASURED
        # tokens instead of an estimate.
        ctx_tokens = token_counter.count(context) if token_counter is not None else None

        rows.append(QAResult(
            qa_id=item.qa_id, category=item.category, abstention=item.abstention,
            has_gold_evidence=has_gold, recall_at_k=r_at_k, mrr=rr, ndcg_at_k=ndcg,
            token_f1=tf1, exact_match=em, bleu1=b1, judged_correct=judged,
            latency_s=latency, total_latency_s=latency + answer_latency,
            answer_latency_s=answer_latency,
            input_tokens=in_tok, output_tokens=out_tok,
            prompt_input_tokens=in_tok, context_tokens=ctx_tokens,
            retrieved_ids=card_ids, retrieved_turn_ids=retrieved_ids,
            answer_text=answer_text,
            question=item.question if not recall_only else None,
            primary_gold=item.primary_gold if not recall_only else None,
        ))
        if on_question is not None:
            on_question()
    return rows, storage, ingested


def _is_rate_limit(exc: BaseException) -> bool:
    """True if ``exc`` looks like an Anthropic rate-limit (429), matched by class
    name / status so the harness needn't import ``anthropic``."""
    name = type(exc).__name__
    return "RateLimit" in name or getattr(exc, "status_code", None) == 429


def run_benchmark(
    conversations: Sequence[Conversation],
    memory: Any,
    *,
    answer_fn: AnswerFn | None = None,
    judge_fn: JudgeFn | None = None,
    judge_model: str | None = None,
    answer_model: str | None = None,
    k: int = DEFAULT_K,
    recall_only: bool = False,
    persona: str = "researcher",
    benchmark: str = "unknown",
    backend: str = "engine",
    sample_fraction: float = 1.0,
    workers: int = 1,
    reuse_ingest: bool = False,
    update_hits: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    count_tokens: bool = True,
) -> dict:
    """Run the full ingest→recall→answer→judge→score loop. Returns a results dict.

    ``judge_model`` / ``answer_model`` select the judge and answer models by
    friendly NAME (haiku / sonnet / opus) or raw id; ``None`` uses the pinned
    reproducibility defaults (``JUDGE_MODEL`` / ``ANSWER_MODEL``), so omitting
    them reproduces the frozen baseline. The RESOLVED ids are recorded in the
    results-JSON config (reproducibility — the whole point of the pin). Only the
    MODEL is user-selected; the judge/answer PROMPT versions stay fixed. When a
    real run (``recall_only=False``) is not given an explicit ``answer_fn`` /
    ``judge_fn``, one is built from the resolved model — so the recorded model
    always matches the model actually used. A stub-injected fn is used as-is
    (dry-run), with the requested model still recorded.

    ``sample_fraction`` (0, 1] evaluates only a deterministic sub-sample of the QA
    items (see ``select_qa_fraction``) for quick iteration. ``workers`` runs the
    per-conversation pipeline across that many threads; it is clamped to
    ``MAX_WORKERS``. Parallelism changes speed only, never the aggregate metrics —
    the sample is fixed up-front and results are reassembled in input order.

    ``reuse_ingest`` skips the ingest step for any conversation whose scope already
    holds memories (detected by a cheap ``memory.count_facts`` — see
    ``_run_conversation``), so recall-only metric iterations run without re-ingesting;
    recall/metrics are byte-identical to a fresh-ingest run over the same memories.
    Conversations with no prior memories are ingested anyway (never silent-empty), and
    the config reports how many were reused vs. (re)ingested.

    ``update_hits`` DEFAULTS TO FALSE here, unlike the engine's own
    ``recall_facts`` (whose True default is the faithful production behavior). The
    harness is a measurement instrument, and the hit writeback makes one question's
    ranking depend on the questions asked before it in the same conversation scope
    (see ``_run_conversation``) — an order effect that contaminates any comparison.
    The value used is recorded in the results config, so a run always states which
    mode produced it.

    ``progress_cb`` (if given) is invoked ``(done, total)`` as each question completes;
    increments are serialized under a lock so it is safe to render a progress bar from
    a parallel (``workers>1``) run. It is a no-op when None."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    # Resolve model names -> ids once, for recording AND (when not injected) for
    # building the callables — a single source of truth so the recorded model
    # can never drift from the model actually queried.
    resolved_answer = resolve_model(answer_model, ANSWER_MODEL)
    resolved_judge = resolve_model(judge_model, JUDGE_MODEL)
    if not recall_only:
        if answer_fn is None:
            from .llm import make_answer_fn  # noqa: PLC0415 — lazy: keeps anthropic optional
            answer_fn = make_answer_fn(resolved_answer)
        if judge_fn is None:
            from .judge import make_judge_fn  # noqa: PLC0415 — lazy: keeps anthropic optional
            judge_fn = make_judge_fn(resolved_judge)

    conversations = select_qa_fraction(conversations, sample_fraction)

    # Measured memory tokens (see token_counter.py). Counted with the ANSWER
    # model's tokenizer, because the answerer is what the memory block is
    # actually paid for. Skipped on a stub dry-run, which has no real answerer
    # and must stay offline.
    counter = TokenCounter(resolved_answer) if (count_tokens and backend != "stub") else None
    envelope = counter.envelope_tokens() if counter is not None else None

    resolved_workers = min(workers, MAX_WORKERS)
    if workers > MAX_WORKERS:
        print(f"! workers={workers} exceeds MAX_WORKERS={MAX_WORKERS}; clamping to "
              f"{MAX_WORKERS} to respect Anthropic rate limits and the DB pool.")

    total_questions = sum(len(conv.qa) for conv in conversations)
    _progress_lock = threading.Lock()
    _done = 0

    def _tick() -> None:
        if progress_cb is None:
            return
        nonlocal _done
        # Increment AND emit under the lock so the callback sees a monotonic
        # (done, total) even when several worker threads finish questions at once.
        with _progress_lock:
            _done += 1
            progress_cb(_done, total_questions)

    def _work(conv: Conversation) -> tuple[list[QAResult], int, bool]:
        return _run_conversation(
            conv, memory, answer_fn=answer_fn, judge_fn=judge_fn,
            k=k, recall_only=recall_only, persona=persona,
            reuse_ingest=reuse_ingest, update_hits=update_hits, on_question=_tick,
            token_counter=counter,
        )

    try:
        if resolved_workers == 1 or len(conversations) <= 1:
            per_conv = [_work(conv) for conv in conversations]
        else:
            with ThreadPoolExecutor(max_workers=resolved_workers) as pool:
                # map() preserves input order -> results are deterministic
                # regardless of worker count.
                per_conv = list(pool.map(_work, conversations))
    except Exception as exc:  # noqa: BLE001 — re-raised with clearer guidance below
        if _is_rate_limit(exc):
            raise RuntimeError(
                "Anthropic rate limit hit during a parallel answer/judge run. "
                f"Lower --workers (was {resolved_workers}) or --fraction and retry."
            ) from exc
        raise

    results: list[QAResult] = []
    total_storage = 0
    n_ingested = 0
    for rows, storage, ingested in per_conv:
        results.extend(rows)
        total_storage += storage
        n_ingested += 1 if ingested else 0
    n_reused = len(conversations) - n_ingested

    summary = _summarize(results, total_storage)
    # Systems provenance: which machine, which answerer, how much parallelism.
    # These belong NEXT TO the latency/token numbers they qualify — a p50 in a
    # file that does not say "workers=4 on this box" is not reproducible, and
    # workers>1 is exactly what makes a reported latency a THROUGHPUT figure
    # rather than a single-stream one (latency_replay.py re-measures at
    # workers=1 for the number that is comparable to a published table).
    summary["machine"] = describe_machine()
    summary["workers"] = resolved_workers
    summary["answer_model"] = None if recall_only else resolved_answer
    summary["token_counting"] = {
        **(counter.stats() if counter is not None else {"available": False}),
        "envelope_tokens": envelope,
        "method": ("anthropic messages.count_tokens on the memory-excerpt block, "
                   "answer model tokenizer, cached by sha256, outside all timed regions"),
    }
    config = {
        "benchmark": benchmark,
        "backend": backend,  # 'engine' = real memory engine; 'stub' = dry-run, NOT a reported score
        "k": k,
        "recall_only": recall_only,
        # Models are USER-SELECTED (recorded here for reproducibility); the
        # answer/judge PROMPT versions are FIXED (pinned + published).
        "answer_model": None if recall_only else resolved_answer,
        "answer_model_pinned_default": ANSWER_MODEL,
        "answer_prompt_version": None if recall_only else ANSWER_PROMPT_VERSION,
        "judge_model": None if recall_only else resolved_judge,
        "judge_model_pinned_default": JUDGE_MODEL,
        "judge_prompt_version": None if recall_only else JUDGE_PROMPT_VERSION,
        "prompt_versions_pinned": True,
        "recall_params": {"rrf_k": RRF_K, "weights": list(RRF_WEIGHTS),
                          "half_life_days": HALF_LIFE_DAYS},
        "sample_fraction": sample_fraction,
        "sample_seed": None if sample_fraction == 1.0 else SAMPLE_SEED,
        "workers": resolved_workers,
        "reuse_ingest": reuse_ingest,
        # Recorded because it changes whether retrieval scores are order-independent
        # (False) or carry a cross-question usage-feedback effect (True).
        "update_hits": update_hits,
        # When reuse_ingest is on: how many conversations reused prior memories vs.
        # were (re)ingested because their scope was empty (the "ingest anyway" note).
        "n_conversations_reused": n_reused,
        "n_conversations_ingested": n_ingested,
        "n_conversations": len(conversations),
        "n_questions": len(results),
    }
    return {"config": config, "summary": summary, "results": [asdict(r) for r in results]}


def run_many(specs: list[dict], max_parallel: int = 2) -> list[dict]:
    """Run several ``run_benchmark`` configs concurrently — for evaluating multiple
    datasets/sets in one pass. Each ``spec`` is a dict of ``run_benchmark`` keyword
    args (it MUST carry ``conversations`` and ``memory``, plus any of ``benchmark`` /
    ``sample_fraction`` / ``workers`` / ``reuse_ingest`` / ``recall_only`` / …).
    Returns one results dict per spec, in input order (``pool.map`` is order-stable),
    regardless of finish order.

    Concurrency guard (the important part): total worker threads across the runs that
    can be in flight at once must stay within ``MAX_WORKERS`` so we never exhaust the
    engine's DB pool (~15 conns) or blow Anthropic rate limits. At most
    ``active = min(max_parallel, len(specs))`` runs execute simultaneously, and each
    run uses up to its own (clamped) ``workers`` threads, so the worst case is the
    ``active`` LARGEST per-spec worker counts running together::

        peak_total_workers = sum(sorted(workers_i, desc)[:active])   # must be <= MAX_WORKERS

    If that peak exceeds ``MAX_WORKERS`` we raise ``ValueError`` up front with a clear
    message (lower ``max_parallel`` or per-spec ``workers``) rather than risk a hang
    or 429/pool-exhaustion mid-run."""
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    specs = list(specs)
    if not specs:
        return []

    active = min(max_parallel, len(specs))
    # Per-spec worker count as run_benchmark will resolve it (default 1, clamped).
    per_workers = sorted(
        (min(max(int(s.get("workers", 1) or 1), 1), MAX_WORKERS) for s in specs),
        reverse=True,
    )
    peak = sum(per_workers[:active])
    if peak > MAX_WORKERS:
        raise ValueError(
            f"run_many concurrency cap exceeded: up to {active} runs can co-run with "
            f"worker counts summing to {peak}, over MAX_WORKERS={MAX_WORKERS}. Lower "
            f"max_parallel or per-spec workers so the peak total stays within the DB "
            f"pool and Anthropic rate limits."
        )

    if active == 1:
        return [run_benchmark(**spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=active) as pool:
        return list(pool.map(lambda spec: run_benchmark(**spec), specs))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarize(results: Sequence[QAResult], storage_bytes: int) -> dict:
    recalls = [r.recall_at_k for r in results if r.recall_at_k is not None]
    mrrs = [r.mrr for r in results if r.has_gold_evidence]
    ndcgs = [r.ndcg_at_k for r in results if r.has_gold_evidence]
    f1s = [r.token_f1 for r in results if r.token_f1 is not None]
    ems = [r.exact_match for r in results if r.exact_match is not None]
    judged = [r.judged_correct for r in results if r.judged_correct is not None]

    per_cat: dict[str, dict[str, float]] = {}
    by_cat: dict[str, list[QAResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat, rows in sorted(by_cat.items()):
        cat_recall = [r.recall_at_k for r in rows if r.recall_at_k is not None]
        cat_judged = [r.judged_correct for r in rows if r.judged_correct is not None]
        cat_f1 = [r.token_f1 for r in rows if r.token_f1 is not None]
        cat_b1 = [r.bleu1 for r in rows if r.bleu1 is not None]
        per_cat[cat] = {
            "n": len(rows),
            "recall_at_k": round(_mean(cat_recall), 4) if cat_recall else None,
            "judge_accuracy": round(judge_accuracy(cat_judged), 4) if cat_judged else None,
            # F1 / B1 / J per question type -- the shape of the mem0 comparison
            # table, so our rows drop straight into it.
            "token_f1": round(_mean(cat_f1), 4) if cat_f1 else None,
            "bleu1": round(_mean(cat_b1), 4) if cat_b1 else None,
        }

    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)
    # ── memory tokens, MEASURED ──────────────────────────────────────────────
    # mem0 Table 2's unit: "the number of tokens extracted during retrieval that
    # serve as context for answering queries", per question. Ours is now the
    # mean of the per-question exact count of the memory-excerpt block, not
    # ``mean(input_tokens) - 87``. Both are reported so a file produced before
    # this change and one produced after can be compared and the size of the
    # old approximation error is visible rather than assumed.
    ctx = [r.context_tokens for r in results if r.context_tokens is not None]
    return {
        "recall_at_k": round(_mean(recalls), 4) if recalls else None,
        "mrr": round(_mean(mrrs), 4) if mrrs else None,
        "ndcg_at_k": round(_mean(ndcgs), 4) if ndcgs else None,
        "token_f1": round(_mean(f1s), 4) if f1s else None,
        "exact_match": round(_mean(ems), 4) if ems else None,
        "bleu1": round(_mean([r.bleu1 for r in results if r.bleu1 is not None]), 4)
                 if any(r.bleu1 is not None for r in results) else None,
        "judge_accuracy": round(judge_accuracy(judged), 4) if judged else None,
        "mean_latency_s": round(_mean([r.latency_s for r in results]), 4) if results else None,
        # Latency split the way the mem0 table reports it: SEARCH is retrieval
        # alone, TOTAL adds answer generation. A mean hides the tail, and the
        # tail is what a user feels, so p50/p95 travel with it.
        "latency_search": metrics.percentiles([r.latency_s for r in results]) if results else {},
        "latency_total": metrics.percentiles([r.total_latency_s for r in results]) if results else {},
        # The answer-generation half of `latency_total`, on its own — so a
        # reader can see how much of "total" is the answerer (a different model
        # from mem0's gpt-4o-mini) and how much is the memory system.
        "latency_answer": metrics.percentiles([r.answer_latency_s for r in results])
                          if results else {},
        "mean_total_latency_s": round(_mean([r.total_latency_s for r in results]), 4)
                                if results else None,
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        # The headline token column. None (not a guess) when nothing was counted.
        "memory_tokens": round(_mean(ctx), 1) if ctx else None,
        "memory_tokens_n_counted": len(ctx),
        "memory_tokens_pct": metrics.percentiles(ctx) if ctx else {},
        "context_tokens_total": sum(ctx) if ctx else 0,
        # Mean answer-prompt input tokens as billed by the API — the quantity the
        # retired estimate ``mean_prompt_input_tokens - 87`` was derived from.
        # Kept so the two are auditable side by side.
        "mean_prompt_input_tokens": round(in_tok / len(results), 1) if results else None,
        "est_cost_usd": round(metrics.token_cost_usd(in_tok, out_tok), 6),
        "storage_bytes": storage_bytes,
        "per_category": per_cat,
    }


def print_summary(results: dict) -> None:
    """Print a compact human-readable summary table."""
    cfg, s = results["config"], results["summary"]
    print(f"\n=== {cfg['benchmark']} — {cfg['n_questions']} questions, "
          f"{cfg['n_conversations']} conversations, k={cfg['k']} ===")
    if cfg.get("sample_fraction", 1.0) != 1.0 or cfg.get("workers", 1) != 1:
        print(f"  run: fraction={cfg.get('sample_fraction')} (seed={cfg.get('sample_seed')})  "
              f"workers={cfg.get('workers')}")
    if cfg.get("reuse_ingest"):
        note = f"  reuse-ingest: reused {cfg.get('n_conversations_reused', 0)} conversation(s)"
        reing = cfg.get("n_conversations_ingested", 0)
        if reing:
            note += (f"; ingested {reing} anyway (no prior memories — recall "
                     f"would be empty otherwise)")
        print(note)
    if not cfg["recall_only"]:
        print(f"  answer={cfg['answer_model']} ({cfg['answer_prompt_version']})  "
              f"judge={cfg['judge_model']} ({cfg['judge_prompt_version']})")
        print("  (model user-selected; prompt versions pinned — default judge "
              f"pin={cfg['judge_model_pinned_default']})")
    print("  retrieval:  recall@k={recall_at_k}  MRR={mrr}  nDCG@k={ndcg_at_k}".format(**s))
    if not cfg["recall_only"]:
        print("  answer:     token-F1={token_f1}  EM={exact_match}  judge-acc={judge_accuracy}".format(**s))
    print(f"  systems:    mean-latency={s['mean_latency_s']}s  tokens(in/out)="
          f"{s['total_input_tokens']}/{s['total_output_tokens']}  "
          f"~${s['est_cost_usd']}  storage={s['storage_bytes']}B")
    tc = s.get("token_counting") or {}
    print(f"  latency:    search p50/p95={s['latency_search'].get('p50')}/"
          f"{s['latency_search'].get('p95')}  total p50/p95="
          f"{s['latency_total'].get('p50')}/{s['latency_total'].get('p95')}  "
          f"(workers={s.get('workers')})")
    print(f"  memory tokens/question (MEASURED, count_tokens): {s.get('memory_tokens')} "
          f"over {s.get('memory_tokens_n_counted')} questions; envelope="
          f"{tc.get('envelope_tokens')}; mean prompt input="
          f"{s.get('mean_prompt_input_tokens')}")
    m = s.get("machine") or {}
    print(f"  machine:    {m.get('cpu_name') or m.get('processor')} | "
          f"{m.get('ram_gb')}GB | {(m.get('gpu') or {}).get('name')} | "
          f"emb={m.get('embedding_backend')} db={m.get('db_name')}")
    print("  per-category:")
    for cat, m in s["per_category"].items():
        print(f"    {cat:22s} n={m['n']:<4d} recall@k={m['recall_at_k']}  judge-acc={m['judge_accuracy']}")
