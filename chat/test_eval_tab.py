"""Headless tests for the Evaluation tab + API-key persistence.

The eval-tab wiring is proved with the dependency-free ``StubMemoryClient`` and
the shipped stub answer/judge callables over the tiny real-format LoCoMo fixture —
no Postgres, no ONNX, no API key. It exercises the SAME ``run_eval`` the UI calls,
so the reuse of Qiyas's ``eval.harness.run_benchmark`` + adapters + metrics is
covered end-to-end. The key-persistence tests write to a temp path and NEVER use a
real key.
"""
from __future__ import annotations

from pathlib import Path

from eval._runner import _stub_answer_fn, _stub_judge_fn
from eval.memory_client import StubMemoryClient

from chat.app import on_save_key, persist_api_key, render_debug
from chat.engine import RecalledFact, TurnResult
from chat.eval_tab import (
    _RB_PARAMS,
    DatasetMissing,
    _build_spec,
    fetch_hint,
    load_conversations,
    render_results,
    run_eval,
    run_eval_many,
)
from eval.memory_client import DEFAULT_K

TINY_LOCOMO = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_locomo.json"
_FAKE_KEY = "sk-ant-TESTONLY-not-a-real-key"  # never a real credential


def _convs():
    return load_conversations("LoCoMo", TINY_LOCOMO)


# ── eval-tab wiring (stub backend, tiny fixture) ────────────────────────────────

def test_recall_only_run_scores_retrieval_and_writes_json(tmp_path):
    """recall-only sweep ($0): reuses run_benchmark over the stub, produces the
    retrieval metrics, and persists a results JSON the UI cites."""
    out = tmp_path / "locomo_results.json"
    run = run_eval(
        "LoCoMo", sample_fraction=1.0, workers=1, recall_only=True, use_judge=False,
        conversations=_convs(), memory=StubMemoryClient(), out_path=out,
    )
    s, cfg = run.results["summary"], run.results["config"]
    assert run.recall_only is True
    assert cfg["n_conversations"] == 2 and cfg["n_questions"] == 7
    assert s["recall_at_k"] == 1.0  # stub retrieves every gold turn
    assert s["token_f1"] is None  # recall-only: no answer scoring
    assert out.exists() and run.out_path == out
    # The rendered dashboard is well-formed HTML citing the JSON path.
    html_out = render_results(run)
    assert "recall@k" in html_out and str(out) in html_out


def test_full_stub_pipeline_runs_answer_and_judge(tmp_path):
    """Full ingest→recall→answer→judge→score with the shipped stub answer/judge —
    proves the answer/judge plumbing flows through run_eval without an API key."""
    run = run_eval(
        "LoCoMo", sample_fraction=1.0, workers=1, recall_only=False, use_judge=True,
        conversations=_convs(), memory=StubMemoryClient(),
        answer_fn=_stub_answer_fn(), judge_fn=_stub_judge_fn(),
        out_path=tmp_path / "r.json",
    )
    s = run.results["summary"]
    assert run.recall_only is False and run.judge_downgraded is False
    assert s["token_f1"] is not None and s["judge_accuracy"] is not None
    assert s["total_input_tokens"] > 0


def test_judge_off_downgrades_to_recall_only(tmp_path):
    """LLM-judge OFF collapses to the $0 recall-only sweep (the harness has no
    answer-without-judge mode) and the downgrade is surfaced, not hidden."""
    run = run_eval(
        "LoCoMo", sample_fraction=1.0, workers=1, recall_only=False, use_judge=False,
        conversations=_convs(), memory=StubMemoryClient(), out_path=tmp_path / "r.json",
    )
    assert run.recall_only is True and run.judge_downgraded is True
    assert "recall-only" in render_results(run).lower()


def test_sample_fraction_passes_through_or_falls_back(tmp_path):
    """The % control reaches the harness when supported; on a build without the
    param it falls back to a local conversation-level slice — so it always bites."""
    run = run_eval(
        "LoCoMo", sample_fraction=0.5, workers=2, recall_only=True, use_judge=False,
        conversations=_convs(), memory=StubMemoryClient(), out_path=tmp_path / "r.json",
    )
    if "sample_fraction" in _RB_PARAMS:
        assert run.results["config"]["n_conversations"] == 2  # harness owns sampling
    else:
        assert run.results["config"]["n_conversations"] == 1  # local fallback sliced
    assert run.fraction_pct == 50 and run.workers == 2


def test_missing_dataset_raises_with_fetch_hint(tmp_path):
    """A benchmark that has not been fetched raises DatasetMissing carrying the
    exact fetch command (the UI renders it instead of crashing)."""
    try:
        run_eval("LongBench", sample_fraction=1.0, workers=1, recall_only=True,
                 use_judge=False, root=tmp_path)
    except DatasetMissing as exc:
        assert "datasets/fetch.py longbench" in str(exc)
    else:
        raise AssertionError("expected DatasetMissing for an unfetched dataset")
    assert fetch_hint("LongBench").endswith("datasets/fetch.py longbench")


# ── reuse-ingest toggle, progress bar, multi-run (the new upgrades) ─────────────

class _FakeProgress:
    """Stand-in for gr.Progress(): records every ``progress(frac, desc=…)`` call."""

    def __init__(self):
        self.calls: list[tuple[float, str]] = []

    def __call__(self, frac: float, desc: str = "") -> None:
        self.calls.append((frac, desc))


def test_reuse_ingest_toggle_skips_reingest(tmp_path):
    """Reuse-ingest passes ``reuse_ingest`` to the harness: a second run over the
    same stub scope reuses the prior ingest (bucket does not grow) — the fast
    recall-only iteration path."""
    stub = StubMemoryClient()
    kw = dict(sample_fraction=1.0, workers=1, recall_only=True, use_judge=False,
              conversations=_convs(), memory=stub)
    fresh = run_eval("LoCoMo", **kw, out_path=tmp_path / "a.json")
    bucket_sizes = {p: len(f) for p, f in stub._store.items()}

    reuse = run_eval("LoCoMo", reuse_ingest=True, **kw, out_path=tmp_path / "b.json")
    if "reuse_ingest" in _RB_PARAMS:
        assert fresh.results["config"]["reuse_ingest"] is False
        assert reuse.results["config"]["reuse_ingest"] is True
        assert reuse.results["config"]["n_conversations_reused"] == 2
        # Ingest skipped: no stub bucket grew (a re-ingest would have doubled them).
        assert {p: len(f) for p, f in stub._store.items()} == bucket_sizes
    # Metrics identical either way — reuse must not change the score.
    assert reuse.results["summary"]["recall_at_k"] == fresh.results["summary"]["recall_at_k"]


def test_progress_bar_driven_by_harness_cb(tmp_path):
    """The Gradio progress bar is driven by the harness ``progress_cb(done,total)`` —
    it advances once per question and reaches 1.0 (no recall-counting proxy)."""
    prog = _FakeProgress()
    run_eval("LoCoMo", sample_fraction=1.0, workers=1, recall_only=True, use_judge=False,
             conversations=_convs(), memory=StubMemoryClient(), out_path=tmp_path / "r.json",
             progress=prog)
    assert prog.calls, "progress was never called"
    if "progress_cb" in _RB_PARAMS:
        fractions = [f for f, _ in prog.calls]
        assert max(fractions) == 1.0  # completes
        # 7 questions in the fixture -> at least 7 per-question ticks after ingest.
        assert len(prog.calls) >= 7


def test_run_eval_many_two_benchmarks_order_stable(tmp_path):
    """run_eval_many runs multiple selected benchmarks concurrently (via run_many)
    and returns one EvalRun per benchmark, in input order, each with its own JSON."""
    convs = _convs()
    runs = run_eval_many(
        ["LoCoMo", "LongMemEval"],
        sample_fraction=1.0, workers=2, recall_only=True, use_judge=False, max_parallel=2,
        conversations_map={"LoCoMo": convs, "LongMemEval": convs},
        memory=StubMemoryClient(), root=tmp_path,
    )
    assert [r.benchmark for r in runs] == ["LoCoMo", "LongMemEval"]  # order-stable
    assert all(r.out_path.exists() for r in runs)  # one JSON each
    # Each renders its own results block.
    combined = "\n".join(render_results(r) for r in runs)
    assert combined.count("recall@k") >= 2


# ── judge/answer model dropdowns thread into the harness spec ───────────────────

def test_selected_models_thread_into_run_benchmark_spec():
    """The judge/answer model dropdowns reach run_benchmark: _build_spec forwards
    them (when the installed harness accepts them) so the recorded model is the
    user's choice, not just the pinned default."""
    spec, _meta = _build_spec(
        "LoCoMo", sample_fraction=1.0, workers=1, recall_only=False, use_judge=True,
        reuse_ingest=False, k=DEFAULT_K, conversations=_convs(),
        memory=StubMemoryClient(), answer_fn=_stub_answer_fn(), judge_fn=_stub_judge_fn(),
        out_path=None, root=Path("/tmp"), progress_cb=None,
        answer_model="opus", judge_model="sonnet",
    )
    if "answer_model" in _RB_PARAMS:
        assert spec["answer_model"] == "opus"
    if "judge_model" in _RB_PARAMS:
        assert spec["judge_model"] == "sonnet"


def test_selected_models_recorded_in_results_config(tmp_path):
    """End-to-end: a full stub run with user-selected models records the RESOLVED
    ids in the results config (reproducibility) — surfaced in the rendered header."""
    run = run_eval(
        "LoCoMo", sample_fraction=1.0, workers=1, recall_only=False, use_judge=True,
        conversations=_convs(), memory=StubMemoryClient(),
        answer_fn=_stub_answer_fn(), judge_fn=_stub_judge_fn(),
        answer_model="opus", judge_model="sonnet", out_path=tmp_path / "r.json",
    )
    cfg = run.results["config"]
    # Harness resolves the friendly name -> full id (containing the family name).
    assert "opus" in str(cfg["answer_model"]) and "sonnet" in str(cfg["judge_model"])
    header = render_results(run)
    assert "models user-selected, prompts pinned" in header


# ── chat debug panel renders the production-parity recall fields ────────────────

def test_debug_panel_shows_event_written_salience_reason():
    result = TurnResult(
        recalled=[RecalledFact(
            rank=1, summary="a recalled fact", scope="own", score=1103.0, matched=True,
            hit_count=4, occurred_at="2026-07-10", created_at="2026-07-20",
        )],
        context_block="ctx", reply="ok", llm_used=True, stored={"written": 1, "mode": "gated"},
        turn_text="t",
    )
    out = render_debug(result)
    # Columns present and distinct: event (occurred) vs written (created).
    assert ">event<" in out and ">written<" in out
    assert ">salience<" in out and ">reason<" in out
    assert "2026-07-10" in out and "2026-07-20" in out  # both timestamps surfaced
    assert "1103.0" in out                              # salience value
    assert "match" in out                               # match-vs-recency reason badge
    assert "when the fact happened" in out              # the legend


# ── API-key persistence ─────────────────────────────────────────────────────────

def test_persist_writes_new_key_line(tmp_path):
    env = tmp_path / ".env"
    assert persist_api_key(_FAKE_KEY, env_path=env) is True
    assert f"ANTHROPIC_API_KEY={_FAKE_KEY}" in env.read_text().splitlines()


def test_persist_updates_existing_key_and_preserves_others(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# header\nMEMORY_DATABASE_URL=postgresql://x\nANTHROPIC_API_KEY=old-value\n")
    persist_api_key(_FAKE_KEY, env_path=env)
    lines = env.read_text().splitlines()
    assert "# header" in lines
    assert "MEMORY_DATABASE_URL=postgresql://x" in lines  # unrelated line preserved
    assert f"ANTHROPIC_API_KEY={_FAKE_KEY}" in lines
    assert sum(1 for line in lines if line.startswith("ANTHROPIC_API_KEY=")) == 1  # updated, not duplicated


def test_persist_empty_value_is_noop(tmp_path):
    env = tmp_path / ".env"
    assert persist_api_key("   ", env_path=env) is False
    assert not env.exists()


def test_save_confirmation_never_echoes_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr("chat.app._ENV_PATH", tmp_path / ".env")
    msg = on_save_key(_FAKE_KEY)
    assert _FAKE_KEY not in msg  # confirmation must not leak the secret
    assert "Saved" in msg
