"""Evaluation tab — run Qiyas's benchmark harness against the real engine.

This is presentation + Gradio glue over the SHIPPED eval harness; it reimplements
no metric and no scoring. The flow reuses, verbatim:

  * ``eval.adapters.{locomo,longmemeval,longbench}.load`` — parse a fetched dataset
    into the common ``Conversation`` format.
  * ``eval.harness.run_benchmark`` — the ingest→recall→answer→judge→score loop that
    produces the results dict (retrieval + answer + systems metrics).
  * ``eval.memory_client.MemoryClient`` — the real ``EngineBackend`` over
    ``MEMORY_DATABASE_URL`` (NOT the stub; the stub is a dry-run only).
  * ``eval.llm.make_answer_fn`` / ``eval.judge.make_judge_fn`` — the pinned answer +
    LLM-judge callables (Haiku-4.5, published ``judge_prompt.txt``).

Run knobs map to the harness's forward params (Qiyas's ``run_benchmark`` /
``run_many``; CLI ``--fraction`` / ``--workers`` / ``--reuse-ingest``):

  * a **% of dataset** slider → ``sample_fraction``,
  * a **parallel workers** count → ``workers``,
  * a **Reuse ingest** toggle → ``reuse_ingest`` (skip re-ingesting a scope that
    already holds memories — fast recall-only iteration; ingests anyway when empty),
  * a **real progress bar** driven by the harness ``progress_cb(done, total)`` — no
    recall-counting proxy; the harness ticks once per completed question,
  * **multi-select benchmarks** run concurrently via ``eval.harness.run_many`` (one
    results block per benchmark), while a single selection keeps the direct
    ``run_benchmark`` path.

``sample_fraction`` / ``workers`` are still passed only when the installed
``run_benchmark`` accepts them; on a pre-integration build the fraction falls back to
a local conversation-level slice so the control still bites.

The cores (``run_eval`` single / ``run_eval_many`` concurrent) are
dependency-injectable — a headless test drives them with ``StubMemoryClient`` + stub
answer/judge over the tiny LoCoMo fixture, needing no DB and no API key.
"""
from __future__ import annotations

import html
import inspect
import json
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from eval.adapters import locomo, longbench, longmemeval, membench
from eval.adapters.base import Conversation
from eval.harness import run_benchmark, run_many
from eval.memory_client import (
    DEFAULT_K,
    HALF_LIFE_DAYS,
    MemoryClient,
    RRF_K,
    RRF_WEIGHTS,
)
from memory.config import MEMORY_LLM_MODELS

_ROOT = Path(__file__).resolve().parent.parent  # memory-research/

# Selectable answer/judge models (friendly names -> ids in memory.config). "haiku"
# is the pinned reproducibility default; only the model is user-selected, the
# answer/judge PROMPT versions stay fixed.
MODEL_CHOICES = list(MEMORY_LLM_MODELS)
DEFAULT_MODEL = "haiku"

# Which run_benchmark params exist in THIS build — so the new sample_fraction /
# workers kwargs are passed only when present (Qiyas's change lands at integration).
_RB_PARAMS = set(inspect.signature(run_benchmark).parameters)


@dataclass(frozen=True)
class _Bench:
    slug: str  # fetch.py argument + results-JSON stem
    load: Callable[[str], Sequence[Conversation]]
    data_rel: str  # default dataset path, relative to _ROOT (mirrors run_<bench>.py)
    license: str


# Benchmark registry. Data paths mirror each eval/run_<bench>.py default_data; the
# licences are the ones passed down (LoCoMo academic-only, the others MIT; v1).
_BENCHMARKS: dict[str, _Bench] = {
    "LoCoMo": _Bench("locomo", locomo.load, "datasets/locomo/locomo10.json", "CC BY-NC 4.0 (academic use only)"),
    "LongMemEval": _Bench("longmemeval", longmemeval.load, "datasets/longmemeval/longmemeval_s_cleaned.json", "MIT"),
    "LongBench": _Bench("longbench", longbench.load, "datasets/longbench", "MIT (LongBench v1)"),
    "MemBench": _Bench("membench", membench.load, "datasets/membench", "MIT (declared; no LICENSE file — fetch-only)"),
}
BENCHMARK_CHOICES = list(_BENCHMARKS)


class DatasetMissing(RuntimeError):
    """Raised when the selected benchmark has not been fetched locally yet."""


def resolve_data_path(benchmark: str, *, root: Path = _ROOT) -> Path:
    return root / _BENCHMARKS[benchmark].data_rel


def fetch_hint(benchmark: str) -> str:
    """The exact command that populates the (gitignored) dataset folder."""
    return f"uv run --extra test python datasets/fetch.py {_BENCHMARKS[benchmark].slug}"


def load_conversations(benchmark: str, path: Path) -> list[Conversation]:
    """Parse a fetched dataset via its shipped adapter (no reimplementation)."""
    return list(_BENCHMARKS[benchmark].load(str(path)))


@dataclass
class EvalRun:
    """A completed run: the harness results dict + where the JSON was written,
    plus the display knobs (mode/fraction/workers) for the reproducibility header."""

    results: dict
    out_path: Path
    benchmark: str
    fraction_pct: int
    workers: int
    recall_only: bool
    judge_downgraded: bool  # judge OFF collapsed a full run to the recall-only sweep


@dataclass
class _SpecMeta:
    """Display knobs for one prepared run — everything ``EvalRun`` needs that the
    harness results dict does not already carry."""

    benchmark: str
    fraction_pct: int
    workers: int
    recall_only: bool
    judge_downgraded: bool
    out_path: Path


def _build_spec(
    benchmark: str,
    *,
    sample_fraction: float,
    workers: int,
    recall_only: bool,
    use_judge: bool,
    reuse_ingest: bool,
    k: int,
    conversations: Sequence[Conversation] | None,
    memory: Any | None,
    answer_fn: Callable | None,
    judge_fn: Callable | None,
    out_path: Path | None,
    root: Path,
    progress_cb: Callable[[int, int], None] | None,
    answer_model: str | None = None,
    judge_model: str | None = None,
) -> tuple[dict, _SpecMeta]:
    """Prepare one ``run_benchmark`` kwargs dict (a ``run_many`` spec) + its display
    meta — shared by the single (``run_eval``) and concurrent (``run_eval_many``)
    paths. Loads the dataset, resolves the backend + pinned answer/judge callables,
    and forwards the run knobs the installed harness accepts.
    """
    bench = _BENCHMARKS[benchmark]

    if conversations is None:
        path = resolve_data_path(benchmark, root=root)
        if not path.exists():
            raise DatasetMissing(
                f"{benchmark} dataset not found at {path}. Fetch it first:\n    {fetch_hint(benchmark)}"
            )
        conversations = load_conversations(benchmark, path)
    conversations = list(conversations)

    # LLM-judge OFF has no "answer-without-judge" mode in the harness, so it
    # collapses to the $0 recall-only sweep — surfaced to the user, not hidden.
    effective_recall_only = recall_only or not use_judge
    judge_downgraded = (not recall_only) and (not use_judge)

    # Run knobs: pass only if this build's harness accepts them (forward-compat).
    extra: dict[str, Any] = {}
    if "sample_fraction" in _RB_PARAMS:
        extra["sample_fraction"] = sample_fraction
    elif sample_fraction < 1.0:  # forward-compat fallback: local conversation slice
        n = max(1, math.ceil(sample_fraction * len(conversations)))
        conversations = conversations[:n]
    if "workers" in _RB_PARAMS:
        extra["workers"] = workers
    if "reuse_ingest" in _RB_PARAMS:
        extra["reuse_ingest"] = reuse_ingest
    if progress_cb is not None and "progress_cb" in _RB_PARAMS:
        extra["progress_cb"] = progress_cb
    # User-selected answer/judge models — forwarded when the installed harness
    # accepts them (Qiyas's change). The harness resolves the friendly name -> id,
    # builds the model-bound callables when none is injected, and RECORDS the
    # resolved id in the results config (reproducibility). Only the MODEL is
    # user-selected; the answer/judge PROMPT versions stay pinned.
    if not effective_recall_only:
        if "answer_model" in _RB_PARAMS:
            extra["answer_model"] = answer_model
        if "judge_model" in _RB_PARAMS:
            extra["judge_model"] = judge_model

    if memory is None:
        memory = MemoryClient()
    if not effective_recall_only:
        # New harness (accepts *_model) builds the model-bound answer/judge fns
        # itself, so the recorded model can never drift from the one queried. On an
        # older build without those params, fall back to pre-building the pinned
        # defaults. An explicitly injected fn (a test stub) is always used as-is.
        if answer_fn is None and "answer_model" not in _RB_PARAMS:
            from eval.llm import make_answer_fn

            answer_fn = make_answer_fn()
        if judge_fn is None and "judge_model" not in _RB_PARAMS:
            from eval.judge import make_judge_fn

            judge_fn = make_judge_fn()
    else:
        answer_fn = judge_fn = None

    spec = dict(
        conversations=conversations,
        memory=memory,
        answer_fn=answer_fn,
        judge_fn=judge_fn,
        k=k,
        recall_only=effective_recall_only,
        benchmark=bench.slug,
        backend="engine",
        # A per-benchmark persona keeps each benchmark's conversation scopes disjoint
        # so concurrent run_many specs never collide, and reuse-ingest is scoped to
        # the benchmark it was ingested under.
        persona=f"eval-{bench.slug}",
        **extra,
    )
    meta = _SpecMeta(
        benchmark=benchmark,
        fraction_pct=round(sample_fraction * 100),
        workers=workers,
        recall_only=effective_recall_only,
        judge_downgraded=judge_downgraded,
        out_path=out_path or (root / f"{bench.slug}_results.json"),
    )
    return spec, meta


def _finalize(results: dict, meta: _SpecMeta) -> EvalRun:
    """Persist the results JSON and wrap it + its display meta into an ``EvalRun``."""
    meta.out_path.write_text(json.dumps(results, indent=2, default=str))
    return EvalRun(
        results=results,
        out_path=meta.out_path,
        benchmark=meta.benchmark,
        fraction_pct=meta.fraction_pct,
        workers=meta.workers,
        recall_only=meta.recall_only,
        judge_downgraded=meta.judge_downgraded,
    )


def _single_progress_cb(progress: Any | None) -> Callable[[int, int], None] | None:
    """Drive a Gradio progress bar from the harness ``progress_cb(done, total)``."""
    if progress is None:
        return None

    def _cb(done: int, total: int) -> None:
        progress(done / max(total, 1), desc=f"Question {done}/{total}")

    return _cb


class _MultiProgress:
    """One Gradio progress bar for a concurrent ``run_many`` — aggregates each
    benchmark's ``progress_cb(done, total)`` into an overall fraction. Per-benchmark
    totals only become known on that benchmark's first tick, so the denominator fills
    in as runs start; updates are serialized so the aggregate is consistent across
    the worker threads run_many uses."""

    def __init__(self, progress: Any | None, labels: Sequence[str]):
        self._progress = progress
        self._lock = threading.Lock()
        self._done = dict.fromkeys(labels, 0)
        self._total = dict.fromkeys(labels, 0)

    def cb_for(self, label: str) -> Callable[[int, int], None] | None:
        if self._progress is None:
            return None

        def _cb(done: int, total: int) -> None:
            with self._lock:
                self._done[label] = done
                self._total[label] = total
                agg_done = sum(self._done.values())
                agg_total = sum(self._total.values()) or 1
                self._progress(agg_done / agg_total,
                               desc=f"{agg_done}/{agg_total} questions across "
                                    f"{len(self._total)} benchmarks")

        return _cb


def run_eval(
    benchmark: str,
    *,
    sample_fraction: float,
    workers: int,
    recall_only: bool,
    use_judge: bool,
    reuse_ingest: bool = False,
    k: int = DEFAULT_K,
    conversations: Sequence[Conversation] | None = None,
    memory: Any | None = None,
    answer_fn: Callable | None = None,
    judge_fn: Callable | None = None,
    answer_model: str | None = None,
    judge_model: str | None = None,
    out_path: Path | None = None,
    root: Path = _ROOT,
    progress: Any | None = None,
) -> EvalRun:
    """Run ONE benchmark through ``eval.harness.run_benchmark`` and persist the JSON.

    Everything is injectable so a headless test drives the whole path with a stub
    backend + stub answer/judge over a tiny fixture (no DB, no API key). Left to
    defaults it uses the REAL ``MemoryClient`` (EngineBackend) and the pinned
    answer/judge callables. The progress bar is driven straight from the harness's
    ``progress_cb`` (one tick per completed question) — no recall-counting proxy.
    """
    if progress is not None:
        progress(0.0, desc="Ingesting…")
    spec, meta = _build_spec(
        benchmark, sample_fraction=sample_fraction, workers=workers,
        recall_only=recall_only, use_judge=use_judge, reuse_ingest=reuse_ingest, k=k,
        conversations=conversations, memory=memory, answer_fn=answer_fn,
        judge_fn=judge_fn, out_path=out_path, root=root,
        progress_cb=_single_progress_cb(progress),
        answer_model=answer_model, judge_model=judge_model,
    )
    return _finalize(run_benchmark(**spec), meta)


def run_eval_many(
    benchmarks: Sequence[str],
    *,
    sample_fraction: float,
    workers: int,
    recall_only: bool,
    use_judge: bool,
    reuse_ingest: bool = False,
    max_parallel: int = 2,
    k: int = DEFAULT_K,
    conversations_map: dict[str, Sequence[Conversation]] | None = None,
    memory: Any | None = None,
    answer_fn: Callable | None = None,
    judge_fn: Callable | None = None,
    answer_model: str | None = None,
    judge_model: str | None = None,
    root: Path = _ROOT,
    progress: Any | None = None,
) -> list[EvalRun]:
    """Run SEVERAL benchmarks concurrently via ``eval.harness.run_many`` — one
    ``EvalRun`` per benchmark, in input order. Each benchmark gets its own aggregated
    slice of a single progress bar. ``conversations_map`` (benchmark → conversations)
    lets a headless test inject fixtures per benchmark; left None each dataset loads
    from disk. The harness enforces the total-concurrency cap (``max_parallel`` ×
    per-run ``workers`` ≤ MAX_WORKERS) and raises ``ValueError`` if exceeded.
    """
    multi = _MultiProgress(progress, benchmarks)
    specs: list[dict] = []
    metas: list[_SpecMeta] = []
    for b in benchmarks:
        convs = (conversations_map or {}).get(b)
        spec, meta = _build_spec(
            b, sample_fraction=sample_fraction, workers=workers,
            recall_only=recall_only, use_judge=use_judge, reuse_ingest=reuse_ingest,
            k=k, conversations=convs, memory=memory, answer_fn=answer_fn,
            judge_fn=judge_fn, out_path=None, root=root, progress_cb=multi.cb_for(b),
            answer_model=answer_model, judge_model=judge_model,
        )
        specs.append(spec)
        metas.append(meta)
    results_list = run_many(specs, max_parallel=max_parallel)
    return [_finalize(r, m) for r, m in zip(results_list, metas, strict=True)]


# ── rendering (reuses the chat debug panel's .mr-* design system) ───────────────

# Eval-only classes, composed into the app's shared <style> (app.py) so the whole
# demo reads as one system. Light + dark via prefers-color-scheme (inherited vars).
EVAL_CSS = """
.mr-kpis { display:flex; flex-wrap:wrap; gap:12px; margin:4px 0 8px; }
.mr-kpi { flex:1 1 140px; border:1px solid var(--mr-line); border-radius:12px;
          padding:12px 14px; background:var(--mr-panel); }
.mr-kpi-label { color:var(--mr-muted); font-size:11.5px; text-transform:uppercase;
                letter-spacing:.04em; }
.mr-kpi-value { font-size:26px; font-weight:650; color:var(--mr-fg); margin-top:2px;
                font-variant-numeric:tabular-nums; }
.mr-kpi-sub { color:var(--mr-muted); font-size:11.5px; margin-top:1px; }
.mr-runhead { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:4px; }
.mr-runhead h3 { margin:0; font-size:16px; color:var(--mr-fg); }
.mr-repro { color:var(--mr-muted); font-size:12px; line-height:1.6; margin-top:6px; }
.mr-repro code { background:var(--mr-code); padding:1px 5px; border-radius:4px; }
.mr-warn { border:1px solid var(--mr-recency); background:var(--mr-recency);
           color:var(--mr-fg); border-radius:10px; padding:8px 12px; font-size:12.5px;
           margin:2px 0 8px; }
"""


def _fmt(v: float | None, *, pct: bool = False, suffix: str = "") -> str:
    if v is None:
        return "—"
    return (f"{v * 100:.1f}%" if pct else f"{v:.3f}") + suffix


def cost_warning(recall_only: bool, use_judge: bool) -> str:
    """Short cost/latency banner (empty when the run is free)."""
    if recall_only or not use_judge:
        return ""
    return (
        '<div class="mr-warn">⚠ LLM-judge on: each question makes <b>2 API calls</b> '
        "(answer + judge) against the pinned Haiku-4.5 — this is slower and spends tokens. "
        "Lower the <b>% of dataset</b> for a quick check first.</div>"
    )


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="mr-kpi-sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="mr-kpi"><div class="mr-kpi-label">{html.escape(label)}</div>'
            f'<div class="mr-kpi-value">{value}</div>{sub_html}</div>')


def _metric_table(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='mr-mono'>{val}</td></tr>" for name, val in rows
    )
    return (f'<div class="mr-sec-title">{html.escape(title)}</div>'
            f"<table class='mr-table'><tbody>{body}</tbody></table>")


def _per_category_table(per_cat: dict) -> str:
    if not per_cat:
        return ""
    rows = []
    for cat, m in per_cat.items():
        rows.append(
            f"<tr><td>{html.escape(cat)}</td><td class='mr-mono'>{m['n']}</td>"
            f"<td class='mr-mono'>{_fmt(m.get('recall_at_k'), pct=True)}</td>"
            f"<td class='mr-mono'>{_fmt(m.get('judge_accuracy'), pct=True)}</td></tr>"
        )
    head = ("<div class='mr-sec-title'>Per-category</div>"
            "<table class='mr-table'><thead><tr><th>category</th><th>n</th>"
            "<th>recall@k</th><th>judge-acc</th></tr></thead><tbody>")
    return head + "".join(rows) + "</tbody></table>"


def render_results(run: EvalRun) -> str:
    cfg, s = run.results["config"], run.results["summary"]
    mode = "recall-only ($0 retrieval sweep)" if run.recall_only else "full (answer + LLM-judge)"

    headline = _fmt(s.get("recall_at_k"), pct=True)
    second = (_fmt(s.get("token_f1")) if run.recall_only
              else _fmt(s.get("judge_accuracy"), pct=True))
    second_label = "token-F1" if run.recall_only else "judge accuracy"
    kpis = "".join([
        _kpi("recall@k", headline, f"k={cfg['k']}"),
        _kpi(second_label, second, "retrieval sweep" if run.recall_only else f"judge {cfg.get('judge_model') or ''}"),
        _kpi("mean latency", _fmt(s.get("mean_latency_s"), suffix="s")),
        _kpi("est. cost", f"${s.get('est_cost_usd', 0):.4f}",
             f"{s.get('total_input_tokens', 0):,}→{s.get('total_output_tokens', 0):,} tok"),
    ])

    retrieval = _metric_table("Retrieval", [
        ("recall@k", _fmt(s.get("recall_at_k"), pct=True)),
        ("MRR", _fmt(s.get("mrr"))),
        ("nDCG@k", _fmt(s.get("ndcg_at_k"))),
    ])
    answer = _metric_table("Answer quality", [
        ("token-F1", _fmt(s.get("token_f1"))),
        ("exact-match", _fmt(s.get("exact_match"), pct=True)),
        ("LLM-judge accuracy", _fmt(s.get("judge_accuracy"), pct=True)),
    ])
    systems = _metric_table("Systems", [
        ("mean latency", _fmt(s.get("mean_latency_s"), suffix="s")),
        ("tokens (in / out)", f"{s.get('total_input_tokens', 0):,} / {s.get('total_output_tokens', 0):,}"),
        ("est. cost (USD)", f"${s.get('est_cost_usd', 0):.6f}"),
        ("storage", f"{s.get('storage_bytes', 0):,} B"),
    ])

    rp = cfg.get("recall_params", {})
    weights = "/".join(str(w) for w in rp.get("weights", RRF_WEIGHTS))
    judge_line = (
        "recall-only — no answer/judge model invoked" if run.recall_only else
        f"answer <code>{html.escape(str(cfg.get('answer_model')))}</code> "
        f"(prompt <code>{html.escape(str(cfg.get('answer_prompt_version')))}</code>) · "
        f"judge <code>{html.escape(str(cfg.get('judge_model')))}</code> "
        f"(prompt <code>{html.escape(str(cfg.get('judge_prompt_version')))}</code>) "
        "— models user-selected, prompts pinned"
    )
    downgrade = ('<div class="mr-warn">LLM-judge is OFF — the harness has no '
                 "answer-without-judge mode, so this ran as the $0 recall-only sweep. "
                 "Turn LLM-judge ON for answer-quality scores.</div>") if run.judge_downgraded else ""

    repro = (
        f'<div class="mr-repro">'
        f"recall params: k={cfg['k']} · RRF K={rp.get('rrf_k', RRF_K)} · "
        f"weights {weights} · half-life {rp.get('half_life_days', HALF_LIFE_DAYS)}d<br>"
        f"{judge_line}<br>"
        f"results JSON: <code>{html.escape(str(run.out_path))}</code><br>"
        f"dataset licence: {html.escape(_BENCHMARKS[run.benchmark].license)}"
        f"</div>"
    )

    return f"""<div class="mr-panel">
  <div class="mr-runhead">
    <h3>{html.escape(run.benchmark)}</h3>
    {_badge_line(cfg, run, mode)}
  </div>
  {downgrade}
  <div class="mr-kpis">{kpis}</div>
  {retrieval}
  {answer}
  {systems}
  {_per_category_table(s.get('per_category', {}))}
  {repro}
</div>"""


def _badge_line(cfg: dict, run: EvalRun, mode: str) -> str:
    parts = [
        f"{cfg['n_questions']} questions",
        f"{cfg['n_conversations']} conversations",
        f"{run.fraction_pct}% of dataset",
        f"{run.workers} worker(s)",
        mode,
        "backend=engine",
    ]
    return "".join(f'<span class="mr-badge mr-badge-own">{html.escape(p)}</span>' for p in parts)


# ── Gradio tab ─────────────────────────────────────────────────────────────────

def _placeholder() -> str:
    return ('<div class="mr-panel"><div class="mr-sec-title">Benchmark results</div>'
            '<p class="mr-hint">Pick a benchmark and press <b>Run benchmark</b>. Results '
            "run Qiyas's eval harness (ingest → recall → answer → judge → score) against the "
            "real memory engine and are also written to a results JSON.</p></div>")


def _error_panel(title: str, body_html: str) -> str:
    return (f'<div class="mr-panel"><div class="mr-sec-title">{html.escape(title)}</div>'
            f"{body_html}</div>")


def _run_failed(exc: Exception) -> str:
    return _error_panel(
        "Run failed",
        f'<p class="mr-err">{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>'
        '<p class="mr-hint">Check <code>MEMORY_DATABASE_URL</code> is set and the '
        "engine DB is migrated, and that <code>ANTHROPIC_API_KEY</code> is set for a "
        "full (LLM-judge) run.</p>",
    )


def on_run(benchmarks, fraction_pct: float, workers: float, max_parallel: float,
           reuse_ingest: bool, recall_only: bool, use_judge: bool,
           answer_model: str, judge_model: str,
           progress=gr.Progress()) -> str:
    """Run one or several selected benchmarks. A single selection takes the direct
    ``run_benchmark`` path; multiple run concurrently via ``run_many``, rendered as
    one results block per benchmark."""
    selected = [b for b in (benchmarks or []) if b in _BENCHMARKS]
    if not selected:
        return _error_panel("No benchmark selected",
                            '<p class="mr-hint">Tick at least one benchmark, then press '
                            "<b>Run benchmarks</b>.</p>")
    common = dict(
        sample_fraction=max(0.01, min(1.0, (fraction_pct or 10) / 100)),
        workers=int(workers or 1),
        reuse_ingest=bool(reuse_ingest),
        recall_only=bool(recall_only),
        use_judge=bool(use_judge),
        answer_model=answer_model or None,
        judge_model=judge_model or None,
        progress=progress,
    )
    try:
        if len(selected) == 1:
            runs = [run_eval(selected[0], **common)]
        else:
            runs = run_eval_many(selected, max_parallel=int(max_parallel or 2), **common)
    except DatasetMissing as exc:
        return _error_panel(
            "Dataset not fetched",
            f'<p class="mr-hint">{html.escape(str(exc))}</p>'
            '<p class="mr-hint">Benchmark data lives in the gitignored '
            "<code>datasets/</code> folder and is never committed.</p>")
    except ValueError as exc:  # run_many concurrency-cap guard — actionable message
        return _error_panel(
            "Too much parallelism",
            f'<p class="mr-err">{html.escape(str(exc))}</p>'
            '<p class="mr-hint">Lower <b>Concurrent benchmarks</b> or <b>Parallel '
            "workers</b> so the total stays within the DB pool + API rate limits.</p>")
    except Exception as exc:  # engine/DB/API error — surface, do not crash the app
        return _run_failed(exc)
    return "\n".join(render_results(r) for r in runs)


_REUSE_NOTE = (
    "**Ingest mode** — *Reingest* (default) writes every conversation's turns into a "
    "fresh scope each run: the honest, from-scratch measurement. *Reuse ingest* skips "
    "the write for any scope that already holds memories, so recall-only metric "
    "iterations run fast without re-ingesting (recall + metrics are byte-identical to a "
    "fresh ingest; an empty scope is ingested anyway so recall is never silently empty)."
)


def build_eval_tab() -> None:
    """Build the Evaluation tab contents (call inside a gr.Tab / gr.Blocks)."""
    gr.Markdown("### Evaluation — memory benchmarks")
    gr.Markdown(
        "Runs the shipped eval harness (retrieval + answer-quality + systems metrics) "
        "against the **real** memory engine. Select one benchmark for a direct run, or "
        "several to run them concurrently. Datasets are fetched locally into the "
        "gitignored `datasets/` folder.",
        elem_classes=["mr-subtitle"],
    )
    benchmarks = gr.CheckboxGroup(
        BENCHMARK_CHOICES, value=[BENCHMARK_CHOICES[0]], label="Benchmarks",
        info="One = direct run · multiple = run concurrently (one results block each).",
    )
    with gr.Row():
        fraction = gr.Slider(1, 100, value=10, step=1, label="% of dataset",
                             info="Fraction to run — smaller = faster + cheaper.", scale=3)
        workers = gr.Slider(1, 8, value=4, step=1, label="Parallel workers",
                            info="Threads within a single benchmark run.", scale=2)
        max_parallel = gr.Slider(1, 4, value=2, step=1, label="Concurrent benchmarks",
                                info="How many selected benchmarks run at once.", scale=2)
    with gr.Row():
        reuse_ingest = gr.Checkbox(value=False, label="Reuse ingest (skip re-ingesting)")
        recall_only = gr.Checkbox(value=False, label="Recall-only ($0 retrieval sweep)")
        use_judge = gr.Checkbox(value=True, label="LLM-judge (answer quality)")
    with gr.Row():
        answer_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Answer model",
                                   info="Model that answers each question (full runs only).")
        judge_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Judge model",
                                  info="LLM-judge model (full runs only). Prompts stay pinned.")
    gr.Markdown(_REUSE_NOTE, elem_classes=["mr-subtitle"])
    run_btn = gr.Button("Run benchmarks", variant="primary")
    warn = gr.HTML(value=cost_warning(False, True))
    out = gr.HTML(value=_placeholder())

    def _warn(recall_only: bool, use_judge: bool) -> str:
        return cost_warning(bool(recall_only), bool(use_judge))

    recall_only.change(_warn, [recall_only, use_judge], warn)
    use_judge.change(_warn, [recall_only, use_judge], warn)
    run_btn.click(
        on_run,
        [benchmarks, fraction, workers, max_parallel, reuse_ingest, recall_only,
         use_judge, answer_model, judge_model],
        out,
    )
