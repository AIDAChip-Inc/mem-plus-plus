#!/usr/bin/env python3
"""Run the consolidation-quality τ-sweep eval (see eval/consolidation_eval.py).

Wired like the other benchmark runners (run_locomo / run_longmemeval): a thin CLI
over the reusable ``run_consolidation_eval`` core, with a ``--stub`` dry-run that
needs no engine, DB, or API key.

Examples::

    # real engine + real ONNX embedder (needs MEMORY_DATABASE_URL — see run_demo.sh)
    uv run python -m eval.run_consolidation_eval --out consol_sweep.json

    # offline wiring check (no DB / no key)
    uv run python -m eval.run_consolidation_eval --stub
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .consolidation_eval import (
    DEFAULT_EVAL_K,
    DEFAULT_PERSONA_BASE,
    DEFAULT_SEED,
    DEFAULT_TAUS,
    build_default_corpus,
    make_stub_consolidate,
    print_consolidation_summary,
    run_consolidation_eval,
)
from .memory_client import MemoryClient, StubMemoryClient


def _parse_taus(raw: str) -> list[float]:
    try:
        taus = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --taus {raw!r}: {exc}") from exc
    if not taus or any(not 0.0 < t <= 1.0 for t in taus):
        raise argparse.ArgumentTypeError(f"--taus must be a comma list in (0, 1], got {raw!r}")
    return taus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the consolidation-quality τ-sweep eval.")
    p.add_argument("--taus", type=_parse_taus, default=list(DEFAULT_TAUS),
                   help="comma-separated near-dup thresholds to sweep (default 0.80,0.85,0.90,0.95)")
    p.add_argument("--k", type=int, default=DEFAULT_EVAL_K, help="recall top-k for the snapshot")
    p.add_argument("--n-runs", type=int, default=1,
                   help="runs per τ (mean ±1σ); >1 matters only on the LLM-synthesis path")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="corpus construction seed")
    p.add_argument("--persona-base", default=DEFAULT_PERSONA_BASE,
                   help="base persona; each τ/run uses <base>__t<τ>__r<run>")
    p.add_argument("--stub", action="store_true",
                   help="in-memory stub backend + stub consolidate (no engine, DB, or key)")
    p.add_argument("--out", default=None, help="write the results JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.k < 1 or args.n_runs < 1:
        print("! --k and --n-runs must be >= 1")
        return 2

    if args.stub:
        memory = StubMemoryClient()
        consolidate_fn = make_stub_consolidate(memory)
        backend = "stub"
    else:
        memory = MemoryClient()
        consolidate_fn = None  # default -> real memory.consolidation.consolidate
        backend = "engine"

    corpus = build_default_corpus(args.seed)
    results = run_consolidation_eval(
        memory=memory, consolidate_fn=consolidate_fn, corpus=corpus,
        taus=args.taus, k=args.k, persona_base=args.persona_base,
        n_runs=args.n_runs, seed=args.seed, backend=backend,
    )
    print_consolidation_summary(results)

    out = args.out or "consolidation_eval_results.json"
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
