"""Shared CLI plumbing for the per-benchmark runners.

Keeps ``run_locomo.py`` / ``run_longmemeval.py`` / ``run_longbench.py`` to a
data-path + adapter binding — everything common (args, memory backend
selection, answer/judge wiring, JSON output) lives here.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from .adapters.base import Conversation
from .harness import MAX_WORKERS, print_summary, run_benchmark
from .memory_client import DEFAULT_K, MemoryClient, StubMemoryClient


def build_parser(benchmark: str, default_data: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run the {benchmark} memory benchmark.")
    p.add_argument("--data", default=default_data, help="path to the fetched dataset file/dir")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="recall top-k")
    p.add_argument("--limit", type=int, default=None, help="cap the number of conversations (smoke)")
    p.add_argument("--recall-only", action="store_true",
                   help="retrieval metrics only — skip answer/judge LLM calls ($0)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="evaluate only this fraction (0,1] of QA items — deterministic "
                        "sub-sample for quick iteration (default 1.0 = full)")
    p.add_argument("--workers", type=int, default=1,
                   help=f"parallel workers over conversations (default 1; clamped to "
                        f"{MAX_WORKERS})")
    p.add_argument("--reuse-ingest", action="store_true",
                   help="skip ingest for a conversation whose scope already has memories "
                        "(fast recall-only metric iteration; ingests anyway if empty)")
    p.add_argument("--stub", action="store_true",
                   help="use the in-memory stub backend + echo answer/judge (no engine, no API key)")
    p.add_argument("--judge-model", default=None,
                   help="judge model: a friendly name (haiku|sonnet|opus) or a raw model id; "
                        "default = the pinned reproducibility judge (haiku). The judge PROMPT "
                        "version is fixed regardless — only the model swaps.")
    p.add_argument("--answer-model", default=None,
                   help="answer model: a friendly name (haiku|sonnet|opus) or a raw model id; "
                        "default = the pinned reproducibility answerer (haiku).")
    p.add_argument("--out", default=None, help="write the results JSON here")
    p.add_argument("--no-count-tokens", action="store_true",
                   help="skip the per-question count_tokens call that MEASURES the "
                        "memory-excerpt tokens (it is free of inference cost and runs "
                        "outside every timed region — use only when offline; the "
                        "summary then reports memory_tokens=None rather than a guess)")
    return p


def _stub_answer_fn():
    from .llm import LLMReply

    def answer_fn(prompt: str) -> LLMReply:
        # Deterministic no-LLM answerer: echo the first recalled excerpt line.
        for line in prompt.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                return LLMReply(text=line.split(". ", 1)[1], input_tokens=len(prompt.split()), output_tokens=8)
        return LLMReply(text="I don't know", input_tokens=len(prompt.split()), output_tokens=4)

    return answer_fn


def _stub_judge_fn():
    from .metrics import token_f1

    def judge_fn(prompt: str) -> str:
        # Crude stand-in: CORRECT when generated overlaps gold. For dry-runs only.
        gold = _extract(prompt, "Gold answer:")
        gen = _extract(prompt, "Generated answer:")
        return "CORRECT" if token_f1(gen, gold) >= 0.5 else "INCORRECT"

    return judge_fn


def _extract(prompt: str, label: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(label):
            return line[len(label):].strip()
    return ""


def run(
    *,
    benchmark: str,
    default_data: str,
    loader: Callable[[str], Sequence[Conversation]],
    argv: list[str] | None = None,
) -> int:
    args = build_parser(benchmark, default_data).parse_args(argv)

    if not (0.0 < args.fraction <= 1.0):
        print(f"! --fraction must be in (0, 1], got {args.fraction}")
        return 2
    if args.workers < 1:
        print(f"! --workers must be >= 1, got {args.workers}")
        return 2

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"! dataset not found at {data_path}\n  fetch it: python datasets/fetch.py {benchmark.split('_')[0]}")
        return 2

    conversations = list(loader(str(data_path)))
    if args.limit:
        conversations = conversations[: args.limit]

    if args.stub:
        memory = StubMemoryClient()
        answer_fn, judge_fn = _stub_answer_fn(), _stub_judge_fn()
    else:
        memory = MemoryClient()
        if args.recall_only:
            answer_fn = judge_fn = None
        else:
            from .judge import make_judge_fn
            from .llm import make_answer_fn
            # Build from the selected models (None => pinned default) so the fn
            # and the model recorded by run_benchmark are the same choice.
            answer_fn = make_answer_fn(args.answer_model)
            judge_fn = make_judge_fn(args.judge_model)

    results = run_benchmark(
        conversations, memory, answer_fn=answer_fn, judge_fn=judge_fn,
        judge_model=args.judge_model, answer_model=args.answer_model,
        k=args.k, recall_only=args.recall_only, benchmark=benchmark,
        backend="stub" if args.stub else "engine",
        sample_fraction=args.fraction, workers=args.workers,
        reuse_ingest=args.reuse_ingest,
        count_tokens=not args.no_count_tokens,
    )
    if args.stub:
        print("  (backend=stub — DRY-RUN wiring check, not a reportable score)")
    print_summary(results)

    out = args.out or f"{benchmark}_results.json"
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out}")
    return 0
