"""Consolidation-QUALITY eval — does synthesize-merge shrink the corpus WITHOUT
losing recall?

Consolidation's whole point is to remove near-duplicate distractors while keeping
every distinct fact retrievable. This module measures exactly that trade with a
**τ sweep** (near-dup cosine thresholds): for each τ it snapshots recall@k on a
fixed query set BEFORE ``consolidate(persona, threshold=τ)``, runs the pass, then
re-snapshots recall@k AFTER, and reports:

  * ``pools_merged``          — near-dup groups collapsed into a synthesized parent
  * ``corpus_reduction_pct``  — (memories_before − memories_after) / before × 100
  * ``recall_retention``      — mean recall@k AFTER / BEFORE (the faithfulness number:
                                the synthesized parent must stay retrievable for the
                                merged cluster's queries; retention < 1 means recall lost)
  * ``lost_recall_qids``      — queries whose gold was retrievable BEFORE but not AFTER

**Judge-free headline (metric rigor).** ``recall_retention`` and
``corpus_reduction_pct`` are pure DETERMINISTIC retrieval + row counts — no LLM
judge is involved, so they hold with or without an ``ANTHROPIC_API_KEY``. An LLM
judge would only enter if we scored synthesized-summary *prose* quality (not done
here); the headline never depends on one. Recall is measured by CONTENT NEEDLE
(a verbatim substring the merged cluster's facts share), not by row id — because
consolidation replaces the near-dup children with a NEW synthesized parent row, so
an id-based recall would spuriously read 0 after every merge. A needle survives
into the synthesized parent (the union-of-detail contract), so needle-recall is the
faithful measure of "is this fact still retrievable".

**Reuse (no reimplementation):** recall@k comes from :mod:`eval.metrics`
(``recall_at_k``); multi-run mean±σ from ``metrics.aggregate_runs``; the memory
surface (seed + recall) from :class:`eval.memory_client` (real engine or stub); the
consolidation pass from :func:`memory.consolidation.consolidate` (imported, never
forked). Each τ runs in its OWN persona scope (``<base>__t<τ>__r<run>``) so the
sweep needs no DELETE/reset between thresholds — clean isolation by construction.

**Variance.** With no LLM key the synthesis path is the deterministic lossless
union, and embeddings + greedy grouping are deterministic, so recall/reduction are
structurally reproducible (σ ≈ 0) — a single run suffices. ``n_runs`` is supported
for the LLM-synthesis path (Haiku), where merged prose (hence re-embedding, hence
ranking) can vary; there we report mean ±1σ.
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import metrics
from .memory_client import DEFAULT_K, MemoryClient

# ── Sweep protocol constants ─────────────────────────────────────────────────────
DEFAULT_TAUS: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
# k=10 (NOT the recall_facts default 24): the default corpus is ~58 memories, so a
# k below the corpus size makes recall@k depend on RANKING — the property that
# actually tests whether a synthesized parent stays retrievable under distractor
# pressure. At k >= corpus size everything is retrieved and recall is trivially 1.0.
DEFAULT_EVAL_K = 10
# Corpus construction seed — pins the shuffle so the fixture (and every downstream
# number) is byte-reproducible. Matches Qiyas's frozen-probe seed convention.
DEFAULT_SEED = 1729
DEFAULT_PERSONA_BASE = "consol_eval"

_WORD = re.compile(r"[a-z0-9.]+")


# ── Query fixture ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConsolidationQuery:
    """One recall probe with its gold NEEDLE(s).

    ``gold_needles`` are verbatim substrings that MUST appear in a recalled
    summary for the query to count as a hit — the id-free recall signal that
    survives a synthesize-merge (the union parent carries the needle). ``kind``
    is ``"cluster"`` (targets a near-dup group that consolidation collapses) or
    ``"distractor"`` (targets an untouched singleton — a control that should keep
    recall = 1.0 across the sweep).
    """

    qid: str
    question: str
    gold_needles: tuple[str, ...]
    kind: str


@dataclass
class Corpus:
    """A controlled corpus: verbatim facts to seed + the probe query set.

    ``facts`` is the flat list written via ``store_facts_verbatim``.
    ``n_clusters`` / ``cluster_size`` / ``n_distractors`` describe the KNOWN
    near-dup structure (documented in the results config for reproducibility).
    """

    facts: list[str]
    queries: list[ConsolidationQuery]
    n_clusters: int
    cluster_size: int
    n_distractors: int

    @property
    def descriptor(self) -> dict:
        return {
            "total_memories": len(self.facts),
            "n_clusters": self.n_clusters,
            "cluster_size": self.cluster_size,
            "n_near_dup_memories": self.n_clusters * self.cluster_size,
            "n_distractors": self.n_distractors,
            "n_queries": len(self.queries),
            "n_cluster_queries": sum(1 for q in self.queries if q.kind == "cluster"),
            "n_distractor_queries": sum(1 for q in self.queries if q.kind == "distractor"),
        }


# ── Retention computation (pure — unit-tested) ─────────────────────────────────────
@dataclass(frozen=True)
class RetentionStats:
    """The faithfulness computation over one BEFORE/AFTER recall snapshot pair."""

    mean_recall_before: float
    mean_recall_after: float
    retention: float               # after / before; 1.0 when before == 0 (nothing to lose)
    lost_recall_qids: list[str]    # gold retrievable BEFORE (>0) but NOT after (==0)
    dropped_recall_qids: list[str] = field(default_factory=list)  # after < before (any drop)
    n_scored: int = 0


def compute_retention(
    before: dict[str, float | None], after: dict[str, float | None]
) -> RetentionStats:
    """Recall@k retention over the queries scored in BOTH snapshots.

    A query is scored only where it has gold evidence in both passes (recall not
    None). ``retention`` = mean(after) / mean(before), pinned to 1.0 when nothing
    was retrievable before (0/0 is "nothing lost", not a divide error). A
    ``lost_recall`` case is the key faithfulness failure: gold was retrievable
    before (recall > 0) and is gone after (recall == 0).
    """
    qids = sorted(
        q for q in before
        if before.get(q) is not None and after.get(q) is not None
    )
    b = [before[q] for q in qids]
    a = [after[q] for q in qids]
    mean_b = statistics.fmean(b) if b else 0.0
    mean_a = statistics.fmean(a) if a else 0.0
    retention = (mean_a / mean_b) if mean_b > 0 else 1.0
    lost = [q for q in qids if before[q] > 0 and after[q] == 0]
    dropped = [q for q in qids if after[q] < before[q]]
    return RetentionStats(
        mean_recall_before=mean_b,
        mean_recall_after=mean_a,
        retention=retention,
        lost_recall_qids=lost,
        dropped_recall_qids=dropped,
        n_scored=len(qids),
    )


def reduction_pct(before_count: int, after_count: int) -> float:
    """Corpus reduction %: (before − after) / before × 100. 0.0 for an empty pool."""
    if before_count <= 0:
        return 0.0
    return 100.0 * (before_count - after_count) / before_count


# ── Content-needle recall (reuses metrics.recall_at_k) ─────────────────────────────
def _needles_in_top_k(recalled: Sequence[dict], gold_needles: Sequence[str], k: int) -> list[str]:
    """The gold needles present in the top-k recalled summaries (rank order).

    Truncates to the top-k SUMMARIES first, then collects which gold needles each
    contains (case-insensitive substring). The result feeds ``recall_at_k`` — a
    set-intersection metric, so duplicate hits collapse automatically.
    """
    found: list[str] = []
    for item in recalled[:k]:
        summary = (item.get("summary") or "").lower()
        found += [n for n in gold_needles if n.lower() in summary]
    return found


def recall_for_query(memory, persona: str, query: ConsolidationQuery, k: int) -> float | None:
    """Content-needle recall@k for one query (``update_hits=False`` — a read-only
    eval snapshot must not perturb salience between the BEFORE and AFTER passes)."""
    recalled = memory.recall_facts(persona, query.question, k=k, update_hits=False)
    found = _needles_in_top_k(recalled, query.gold_needles, k)
    return metrics.recall_at_k(query.gold_needles, found, k=None)


# ── Per-τ evaluation ────────────────────────────────────────────────────────────────
def evaluate_tau(
    memory,
    consolidate_fn: Callable[..., dict],
    *,
    persona: str,
    corpus: Corpus,
    queries: Sequence[ConsolidationQuery],
    k: int,
    tau: float,
) -> dict:
    """Seed → recall BEFORE → ``consolidate(persona, threshold=τ)`` → recall AFTER.

    Runs entirely in ``persona``'s own scope (unique per τ/run), so the seeded
    corpus is isolated and no reset is needed. Returns one raw-run row.
    """
    memory.store_facts_verbatim(persona, corpus.facts)

    before = {q.qid: recall_for_query(memory, persona, q, k) for q in queries}
    result = consolidate_fn(persona, threshold=tau)
    after = {q.qid: recall_for_query(memory, persona, q, k) for q in queries}

    ret = compute_retention(before, after)
    mem_before = result["memories_before"]
    mem_after = result["memories_after"]
    return {
        "tau": tau,
        "persona": persona,
        "run_id": result.get("run_id"),
        "pools_merged": result["pools_merged"],
        "superseded": result.get("superseded", 0),
        "memories_before": mem_before,
        "memories_after": mem_after,
        "corpus_reduction_pct": reduction_pct(mem_before, mem_after),
        "recall_at_k_before": ret.mean_recall_before,
        "recall_at_k_after": ret.mean_recall_after,
        "recall_retention": ret.retention,
        "lost_recall_qids": ret.lost_recall_qids,
        "dropped_recall_qids": ret.dropped_recall_qids,
        "n_scored_queries": ret.n_scored,
    }


def _aggregate_tau(tau: float, runs: list[dict]) -> dict:
    """Fold ``n_runs`` raw rows for one τ into a reported row (mean ±1σ where it
    matters). Counts/lost-cases are taken from run 0 (deterministic in the
    judge-free path); reduction% and retention are meaned with their σ so an
    LLM-synthesis run reports its noise band."""
    reduction = metrics.aggregate_runs([r["corpus_reduction_pct"] for r in runs])
    retention = metrics.aggregate_runs([r["recall_retention"] for r in runs])
    before = metrics.aggregate_runs([r["recall_at_k_before"] for r in runs])
    after = metrics.aggregate_runs([r["recall_at_k_after"] for r in runs])
    r0 = runs[0]
    return {
        "tau": tau,
        "n_runs": len(runs),
        "pools_merged": r0["pools_merged"],
        "superseded": r0["superseded"],
        "memories_before": r0["memories_before"],
        "memories_after": r0["memories_after"],
        "corpus_reduction_pct": round(reduction.mean, 2),
        "corpus_reduction_stddev": round(reduction.stddev, 4),
        "recall_at_k_before": round(before.mean, 4),
        "recall_at_k_after": round(after.mean, 4),
        "recall_retention": round(retention.mean, 4),
        "recall_retention_stddev": round(retention.stddev, 4),
        "lost_recall_qids": r0["lost_recall_qids"],
        "dropped_recall_qids": r0["dropped_recall_qids"],
        "n_scored_queries": r0["n_scored_queries"],
    }


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def make_stub_consolidate(memory) -> Callable[..., dict]:
    """A DEPENDENCY-FREE ``consolidate`` double for the ``--stub`` dry-run and unit
    tests (no DB, no engine, no key).

    It mirrors the engine's OFFLINE contract exactly enough to exercise the sweep
    orchestration end-to-end: greedy-disjoint groups a persona's stub facts by
    token-Jaccard ≥ threshold (the stub analogue of ``dedup_groups``' cosine ≥ τ),
    replaces each group with the deterministic lossless union ``" ; ".join`` (the
    same union-of-detail contract as ``deterministic_union_summary``), and returns
    the ``consolidate()`` dict shape. It is a TEST DOUBLE — never a second
    implementation of the real pass — so the stub bucket after a merge still carries
    every needle, and needle-recall behaves like the real no-LLM path.
    """
    def _consolidate(persona: str, *, threshold: float = 0.85, **_ignored) -> dict:
        facts = list(memory._store.get(persona, []))
        before = len(facts)
        seen = [False] * len(facts)
        merged: list[str] = []
        singles: list[str] = []
        for i, fi in enumerate(facts):
            if seen[i]:
                continue
            group = [fi]
            seen[i] = True
            for j in range(i + 1, len(facts)):
                if not seen[j] and _jaccard(fi, facts[j]) >= threshold:
                    group.append(facts[j])
                    seen[j] = True
            if len(group) >= 2:
                merged.append(" ; ".join(group))
            else:
                singles.append(fi)
        memory._store[persona] = singles + merged
        return {
            "run_id": f"stub-{persona}",
            "pools_merged": len(merged),
            "memories_before": before,
            "memories_after": len(memory._store[persona]),
            "superseded": 0,
        }

    return _consolidate


def _default_consolidate() -> Callable[..., dict]:
    """Lazy import of the real engine consolidate (keeps the module importable
    offline / for unit tests, which inject a stub instead)."""
    from memory.consolidation import consolidate  # noqa: PLC0415

    return consolidate


def _synthesis_path() -> str:
    """Which synthesis path consolidate() will take — recorded in config so a
    reader knows whether the run used Haiku union-synthesis or the deterministic
    lossless union (no key). The headline retention/reduction is judge-free either
    way; this only annotates the merged-summary PROSE source."""
    try:
        from memory.llm import llm_available  # noqa: PLC0415

        return "llm-haiku" if llm_available() else "deterministic-union"
    except Exception:
        return "deterministic-union"


def run_consolidation_eval(
    *,
    memory=None,
    consolidate_fn: Callable[..., dict] | None = None,
    corpus: Corpus | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    k: int = DEFAULT_EVAL_K,
    persona_base: str = DEFAULT_PERSONA_BASE,
    n_runs: int = 1,
    seed: int = DEFAULT_SEED,
    backend: str = "engine",
) -> dict:
    """Run the τ-sweep consolidation-quality eval. Returns ``{config, per_tau}``.

    Defaults wire the REAL memory engine + ``memory.consolidation.consolidate``
    and the built-in controlled corpus; tests/dry-runs inject a stub ``memory`` +
    ``consolidate_fn`` and a small ``corpus``. Each (τ, run) gets its own persona
    scope, so the sweep is self-isolating.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    if corpus is None:
        corpus = build_default_corpus(seed)
    if memory is None:
        memory = MemoryClient()
    if consolidate_fn is None:
        consolidate_fn = _default_consolidate()

    per_tau: list[dict] = []
    for tau in taus:
        runs = [
            evaluate_tau(
                memory, consolidate_fn,
                persona=f"{persona_base}__t{tau}__r{run}",
                corpus=corpus, queries=corpus.queries, k=k, tau=tau,
            )
            for run in range(n_runs)
        ]
        per_tau.append(_aggregate_tau(tau, runs))

    config = {
        "eval": "consolidation_quality",
        "backend": backend,  # 'engine' = real memory engine; 'stub' = dry-run, NOT a reportable score
        "taus": list(taus),
        "k": k,
        "n_runs": n_runs,
        "seed": seed,
        "persona_base": persona_base,
        "consolidation_default_tau": _consolidation_default_tau(),
        "synthesis_path": _synthesis_path() if backend == "engine" else "stub",
        "headline_judge_free": True,  # recall@k retention + reduction% are deterministic retrieval
        "recall_metric": "content-needle (id-free; survives synthesize-merge)",
        "corpus": corpus.descriptor,
        "embedding_model": _embedding_model(),
    }
    return {"config": config, "per_tau": per_tau}


def _consolidation_default_tau() -> float:
    try:
        from memory.consolidation import DEFAULT_TAU  # noqa: PLC0415

        return DEFAULT_TAU
    except Exception:
        return 0.85


def _embedding_model() -> str:
    try:
        from memory import config as mcfg  # noqa: PLC0415

        return f"{mcfg.EMBEDDING_MODEL} ({mcfg.EMBEDDING_DIM}d, ONNX)"
    except Exception:
        return "unknown"


# ── Controlled corpus fixture ──────────────────────────────────────────────────────
# Six near-dup CLUSTERS. Each is one underlying fact RESTATED 3 ways (the realistic
# shape of accumulated agent-memory near-dups); all three paraphrases carry the same
# verbatim NEEDLE (a number/id/node the query asks for), so the needle must survive
# the synthesize-merge for recall to be retained. The clusters span a natural
# embedding-similarity gradient (measured with the real ONNX MiniLM: min-pairwise
# cosine ranges ~0.59→0.91), so the τ sweep exercises a spread of near-dup tightness
# rather than a single band — some clusters collapse only at the loose τ, the
# tightest survive to the strict end.
_CLUSTERS: list[tuple[str, str, list[str]]] = [
    # (cluster id, gold needle, [paraphrases])
    ("clock", "1.833 GHz", [
        "clk_main runs at 1.833 GHz.",
        "clk_main runs at 1.833 GHz on the SoC.",
        "The clk_main clock runs at 1.833 GHz.",
    ]),
    ("pll", "PLL-4420", [
        "The clocks are synthesized by the PLL-4420 macro.",
        "The clocks are synthesized by the PLL-4420 hard macro.",
        "On-chip clocks are synthesized by the PLL-4420 macro.",
    ]),
    ("sram", "512 KB", [
        "The shared L2 cache is a 512 KB SRAM.",
        "The shared L2 cache is a 512 KB SRAM block.",
        "Shared L2 is a 512 KB SRAM.",
    ]),
    ("voltage", "0.72 V", [
        "The core supply voltage is 0.72 V nominal.",
        "The core supply voltage is 0.72 V typical.",
        "Core Vdd is 0.72 V nominal.",
    ]),
    ("process", "N5 node", [
        "The chip is fabricated on the TSMC N5 node.",
        "The chip is taped out on the TSMC N5 node.",
        "The die is fabricated on the TSMC N5 node.",
    ]),
    ("thermal", "125 degC", [
        "Tj max is 125 degC for the automotive grade.",
        "Tj max is 125 degC for the auto grade.",
        "Junction temperature max is 125 degC for the automotive grade.",
    ]),
]

# 40 DISTINCT distractor facts — unrelated engineering topics, each a singleton with
# NO cluster needle, so they neither merge with each other nor false-match a cluster
# query. They pad the corpus past k so recall@k depends on ranking.
_DISTRACTORS: list[str] = [
    "The DMA controller supports eight independent scatter-gather channels.",
    "UART0 is configured for a baud rate that the bootloader auto-detects.",
    "The I2C bus is pulled up externally and clocked in fast-mode.",
    "SPI flash holds the first-stage boot image at a fixed offset.",
    "The interrupt controller prioritizes timer interrupts over peripheral ones.",
    "Coverage on the AXI crossbar reached the sign-off goal last week.",
    "The DRC deck was updated to the latest foundry revision.",
    "LVS was clean after the last engineering change order.",
    "The floorplan places the CPU cluster in the north-west corner.",
    "Power gating uses coarse-grained switches on the accelerator domain.",
    "The reset synchronizer uses a two-flop metastability guard.",
    "The scan chain is stitched in physical order to cut routing.",
    "Clock gating cells were inserted automatically during synthesis.",
    "The USB PHY is licensed as a third-party analog hard IP.",
    "The debug bridge exposes a JTAG tap to the on-chip trace buffer.",
    "The memory controller reorders reads ahead of writes when idle.",
    "ECC protects the register file against single-event upsets.",
    "The GPIO block muxes four alternate functions per pin.",
    "The watchdog timer resets the SoC if not serviced in time.",
    "The temperature sensor is calibrated at two reference points.",
    "The bandgap reference trims out process variation at test.",
    "The ADC uses a successive-approximation architecture.",
    "The DAC output buffer drives a fifty-ohm load off chip.",
    "The Ethernet MAC supports jumbo frames up to the configured cap.",
    "The PCIe controller negotiates lane width during link training.",
    "The cache coherency protocol is directory-based across the mesh.",
    "The NoC routers use wormhole flow control with virtual channels.",
    "The security enclave stores keys in a one-time-programmable fuse array.",
    "The random number generator is seeded from ring-oscillator jitter.",
    "The boot ROM verifies the image signature before jumping to it.",
    "The power management unit sequences the rails at cold start.",
    "The retention flops keep state during the deepest sleep mode.",
    "The performance counters expose cache-miss events to software.",
    "The MMU walks a four-level page table on a TLB miss.",
    "The branch predictor uses a two-level adaptive history table.",
    "The vector unit processes lanes in a single-issue pipeline.",
    "The FPU rounds to nearest-even by default per the standard.",
    "The trace encoder compresses the instruction stream on the fly.",
    "The bus fabric arbitrates using a round-robin priority scheme.",
    "The test compression ratio was tuned to fit the tester memory.",
]


def build_default_corpus(seed: int = DEFAULT_SEED) -> Corpus:
    """The built-in controlled corpus: 6 near-dup clusters (×3 paraphrases) + 40
    distinct distractors = 58 memories, with 6 cluster + 4 distractor queries.

    ``seed`` shuffles the seed/write ORDER only (never the content), so the fixture
    is byte-reproducible while write order does not privilege any row's recency.
    """
    import random  # noqa: PLC0415 — local: keep module import side-effect-free

    cluster_size = len(_CLUSTERS[0][2])
    facts: list[str] = []
    queries: list[ConsolidationQuery] = []
    for cid, needle, paraphrases in _CLUSTERS:
        facts.extend(paraphrases)
        queries.append(ConsolidationQuery(
            qid=f"cluster:{cid}",
            question=_CLUSTER_QUESTIONS[cid],
            gold_needles=(needle,),
            kind="cluster",
        ))
    facts.extend(_DISTRACTORS)

    # Four distractor CONTROL queries — untouched singletons whose recall must stay
    # 1.0 across the sweep (consolidation never touches them).
    for qid, question, needle in _DISTRACTOR_PROBES:
        queries.append(ConsolidationQuery(
            qid=qid, question=question, gold_needles=(needle,), kind="distractor",
        ))

    random.Random(seed).shuffle(facts)
    return Corpus(
        facts=facts,
        queries=queries,
        n_clusters=len(_CLUSTERS),
        cluster_size=cluster_size,
        n_distractors=len(_DISTRACTORS),
    )


_CLUSTER_QUESTIONS: dict[str, str] = {
    "clock": "What frequency does the main system clock clk_main run at?",
    "pll": "Which PLL macro synthesizes the on-chip clocks?",
    "sram": "How large is the shared L2 cache SRAM?",
    "voltage": "What is the nominal core supply voltage?",
    "process": "Which foundry process node is the chip fabricated on?",
    "thermal": "What is the maximum junction temperature for the automotive grade?",
}

# (qid, question, gold needle) — each targets exactly ONE distractor fact above.
_DISTRACTOR_PROBES: list[tuple[str, str, str]] = [
    ("distractor:dma", "How many channels does the DMA controller support?", "eight independent scatter-gather channels"),
    ("distractor:ecc", "What protects the register file from single-event upsets?", "ECC"),
    ("distractor:mmu", "How many levels does the page table walk have on a TLB miss?", "four-level page table"),
    ("distractor:rng", "What seeds the random number generator?", "ring-oscillator jitter"),
]


# ── Reporting ──────────────────────────────────────────────────────────────────────
def print_consolidation_summary(results: dict) -> None:
    """Print the τ-sweep table: reduction % + recall@k retention per τ."""
    cfg = results["config"]
    corpus = cfg["corpus"]
    print(f"\n=== consolidation-quality τ-sweep — {corpus['total_memories']} memories "
          f"({corpus['n_clusters']} near-dup clusters ×{corpus['cluster_size']} + "
          f"{corpus['n_distractors']} distractors), k={cfg['k']} ===")
    print(f"  synthesis={cfg['synthesis_path']}  headline judge-free={cfg['headline_judge_free']}  "
          f"embedder={cfg['embedding_model']}  seed={cfg['seed']}  n_runs={cfg['n_runs']}")
    if cfg["backend"] == "stub":
        print("  (backend=stub — DRY-RUN wiring check, NOT a reportable score)")
    header = (f"  {'τ':>5} {'merged':>7} {'before':>7} {'after':>6} {'reduce%':>8} "
              f"{'recall@k↓':>10} {'recall@k↑':>10} {'retention':>10} {'lost':>5}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results["per_tau"]:
        retention = f"{row['recall_retention']:.4f}"
        if row["n_runs"] > 1:
            retention += f"±{row['recall_retention_stddev']}"
        print(f"  {row['tau']:>5.2f} {row['pools_merged']:>7d} {row['memories_before']:>7d} "
              f"{row['memories_after']:>6d} {row['corpus_reduction_pct']:>8.2f} "
              f"{row['recall_at_k_before']:>10.4f} {row['recall_at_k_after']:>10.4f} "
              f"{retention:>10}  {len(row['lost_recall_qids']):>3d}")
    lost = {row["tau"]: row["lost_recall_qids"] for row in results["per_tau"] if row["lost_recall_qids"]}
    if lost:
        print("  LOST-RECALL cases (gold retrievable before, not after):")
        for tau, qids in lost.items():
            print(f"    τ={tau}: {', '.join(qids)}")
    else:
        print("  no lost-recall cases across the sweep (consolidation preserved recall).")
