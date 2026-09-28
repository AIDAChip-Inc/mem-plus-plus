"""System / Architecture tab — the canonical, demo-facing system picture of the
memory-research replica.

Rendered STRICTLY SELF-CONTAINED (no CDN, no mermaid.js): a hand-authored
HTML/CSS *boxes-and-arrows* diagram over the shared ``.mr-*`` design system, then
the system summary and the consolidation/lifecycle ("cron") logic. The HTML/CSS
diagram is the deliberate choice over vendoring mermaid — it is the lightest, it
inherits the exact ``.mr-*`` light/dark theme (so it reads as one system with the
Chat / Evaluation / Memory tabs), and it ships zero third-party JS.

Source of truth: ``docs/design/ARCHITECTURE.md`` (Dina) — every node, constant and
lifecycle claim below mirrors that doc, which is itself verified against the
shipped ``memory/*.py``. This module is PRESENTATION ONLY: it imports nothing from
the engine and needs no DB and no API key, so the tab always renders.
"""
from __future__ import annotations

import gradio as gr

# System-tab-only classes, composed into the app's shared <style> (app.py) so the
# whole demo reads as one system. Reuses the .mr-* vars (light + dark) verbatim; a
# per-subsystem tint (own/write/cons) reuses the existing badge tint vars.
SYS_CSS = """
.mr-archwrap { overflow-x:auto; }
.mr-arch { min-width:720px; display:flex; flex-direction:column; align-items:stretch; gap:0; }
.mr-alayer { display:flex; flex-direction:column; gap:8px; }
.mr-adown { text-align:center; color:var(--mr-muted); font-size:18px; line-height:1;
            margin:2px 0; }
.mr-aflow { display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
.mr-arr { color:var(--mr-accent); font-weight:700; font-size:15px; }
.mr-anode { flex:0 1 auto; border:1px solid var(--mr-line); border-radius:10px;
            padding:8px 11px; background:var(--mr-panel); color:var(--mr-fg);
            font-size:12.5px; min-width:110px; }
.mr-anode-t { font-weight:650; }
.mr-anode-sub { color:var(--mr-muted); font-size:11px; margin-top:2px; line-height:1.45; }
.mr-anode code { background:var(--mr-code); padding:0 4px; border-radius:4px; font-size:11px; }
.mr-agroup { border:1px dashed var(--mr-line); border-radius:12px; padding:10px 12px 12px;
             background:transparent; }
.mr-agroup-t { font-size:11px; font-weight:700; text-transform:uppercase;
               letter-spacing:.04em; color:var(--mr-accent); margin-bottom:8px; }
.mr-agroup-own    { border-color:var(--mr-own);     background:color-mix(in srgb, var(--mr-own) 22%, transparent); }
.mr-agroup-write  { border-color:var(--mr-match);   background:color-mix(in srgb, var(--mr-match) 22%, transparent); }
.mr-agroup-cons   { border-color:var(--mr-recency); background:color-mix(in srgb, var(--mr-recency) 22%, transparent); }
.mr-agroup-store  { border-color:var(--mr-line);    background:var(--mr-code); }
.mr-anote { color:var(--mr-muted); font-size:11px; margin-top:8px; }
.mr-anode-legs { display:flex; flex-wrap:wrap; gap:6px; }
.mr-anode-legs .mr-anode { flex:1 1 150px; }
"""


def _node(title: str, sub: str = "") -> str:
    """One box. ``title``/``sub`` are AUTHORED static HTML (no user input)."""
    sub_html = f'<div class="mr-anode-sub">{sub}</div>' if sub else ""
    return f'<div class="mr-anode"><div class="mr-anode-t">{title}</div>{sub_html}</div>'


def _arr(char: str = "→") -> str:
    return f'<span class="mr-arr">{char}</span>'


def _group(title: str, inner: str, *, kind: str = "") -> str:
    cls = f"mr-agroup mr-agroup-{kind}" if kind else "mr-agroup"
    return (f'<div class="{cls}"><div class="mr-agroup-t">{title}</div>{inner}</div>')


def _flow(*parts: str) -> str:
    return f'<div class="mr-aflow">{"".join(parts)}</div>'


def render_system_diagram() -> str:
    """The whole replica in one self-contained HTML/CSS diagram (no CDN): the three
    consumers, the per-turn PRE→REPLY→POST hook loop, the four-leg recall pipeline
    (→ weighted RRF K=60 w=1/1/4 → MATCH→HITS→RECENCY → recent-reserve → top-24 →
    hit-writeback), the write pipeline (extraction → MiniLM-384 + tsvector → row),
    the consolidation / supersession / merge subsystem, and the Postgres+pgvector
    store — wired so the data flow is legible top-to-bottom."""

    consumers = _flow(
        _node("Chat tab", "chat/app.py · engine.run_turn"),
        _node("Eval harness", "eval/harness.py<br>ingest→recall→answer→judge→score"),
        _node("Memory tab", "browse.py read-only · mutate.py actions"),
    )

    hook = _group(
        "Per-turn hook loop — chat/engine.py::run_turn",
        _flow(
            _node("PRE-hook", "<code>recall_facts(persona, msg, k=24)</code>"),
            _arr(),
            _node("build + inject", "build_context_block → prompt<br>(as additionalContext)"),
            _arr(),
            _node("REPLY", "llm_fn → Haiku / Sonnet / Opus<br>(no key → recall-only degrade)"),
            _arr(),
            _node("POST-hook", "<code>store_facts(…, mode=auto)</code>"),
        ),
    )

    legs = (
        '<div class="mr-anode-legs">'
        + _node("① structured / recent", "created_at DESC<br>(POOL only — not an RRF leg)")
        + _node("② lexical / fuzzy", "websearch_to_tsquery<br>+ ts_rank_cd DESC")
        + _node("③ entity-tag", "normalized query terms<br>vs per-scope registry")
        + _node("④ vector-ANN", "embedding &lt;=&gt; qvec (cosine)<br>MiniLM-384")
        + "</div>"
    )
    recall = _group(
        "Recall pipeline — store.recall_project_sections → _recall_scoped",
        _node("Scope filter FIRST",
              "user_id ∧ agent_type ∧ project_id ∧ is_active<br>"
              "(session_id OMITTED → cross-session)")
        + '<div class="mr-adown">↓</div>'
        + '<div class="mr-agroup-t">4 candidate legs (cap 50 each)</div>'
        + legs
        + '<div class="mr-adown">↓ <span class="mr-anote">legs ②③④ fuse; ① is a pool → rank</span></div>'
        + _flow(
            _node("Weighted RRF", "Σ w/(K+rank) · <b>K=60</b><br>w = <b>1 / 1 / 4</b> (fuzzy/tag/vector)"),
            _arr(),
            _node("Rank", "<b>MATCH → HITS → RECENCY</b><br>decay = 2^(−age_days/7)"),
            _arr(),
            _node("Recent-reserve", "reserve = min(3, k)<br>event-fresh pool"),
            _arr(),
            _node("top-k = 24"),
            _arr(),
            _node("hit-writeback", "batched UPDATE · last_used_at<br>matched only: hit_count+1"),
        )
        + '<div class="mr-anote">Two-section split: <b>OWN</b> scope + <b>TEAM</b> '
          "scope (disjoint) · TEAM cap = 8, OWN backfills k−team.</div>",
        kind="own",
    )

    write = _group(
        "Write pipeline — store_facts → store.write",
        _flow(
            _node("Extraction (Haiku)", "gated (default) | two-section | atomic<br>"
                  "or <code>store_facts_verbatim</code> (LLM-free)"),
            _arr(),
            _node("Embed + index", "MiniLM-384 embedding<br>+ content_tsv (to_tsvector)<br>"
                  "<span class='mr-anode-sub'>agent_summary EXCLUDED</span>"),
            _arr(),
            _node("Insert row", "AgentMemory + reconcile tags<br>valid_from = occurred_at else now()"),
        ),
        kind="write",
    )

    cons = _group(
        "Consolidation / lifecycle — MANUAL on-demand (NOT the prod background loop)",
        _flow(
            _node("consolidate(persona)", "<b>τ = 0.85</b> synthesize-merge<br>+ MemoryConsolidationRun ledger"),
            _arr(),
            _node("dedup_groups", "greedy disjoint<br>cosine ≥ τ, size ≥ 2<br>(drop confidential/pinned)"),
            _arr(),
            _node("classify_group (LLM)", "CONFLICT · RESTATEMENT · DISTINCT"),
        )
        + '<div class="mr-adown">↓</div>'
        + _flow(
            _node("CONFLICT → supersede", "newest-wins · valid_to + superseded_by_id<br>"
                  "<b>keeps is_active = TRUE</b> (queryable)"),
            _node("RESTATEMENT → merge", "union-merge (Haiku MERGE_PROMPT)<br>"
                  "insert_root_archive_sources · is_active = FALSE"),
            _node("DISTINCT → skip", "untouched (safe default)"),
        )
        + '<div class="mr-anote">Manual user ops (Memory tab · mutate.py): '
          "<b>supersede</b> (valid_to + is_active=FALSE) · <b>merge</b> (parent_id forest, "
          "full audit) · <b>reset</b> (soft wipe).</div>",
        kind="cons",
    )

    store = _group(
        "Local store — Postgres 16 + pgvector (docker, host :5433)",
        _flow(
            _node("agent_memory", "Vector(384) HNSW cosine · content_tsv GIN<br>"
                  "bi-temporal: valid_from/valid_to,<br>superseded_by_id, parent_id forest, is_active"),
            _node("memory_tag", "UNIQUE(scope_key, label)"),
            _node("memory_tag_link"),
            _node("memory_consolidation_run", "ledger (upsert per scope_key)"),
        ),
        kind="store",
    )

    def layer(inner: str) -> str:
        return f'<div class="mr-alayer">{inner}</div>'

    down = '<div class="mr-adown">↓</div>'
    body = (
        layer(consumers) + down
        + layer(hook) + down
        + layer(recall) + down
        + layer(write) + down
        + layer(cons) + down
        + layer(store)
    )
    return (
        '<div class="mr-panel"><div class="mr-sec-title">System diagram — the whole '
        "replica in one view</div>"
        '<div class="mr-archwrap"><div class="mr-arch">' + body + "</div></div>"
        '<p class="mr-anote">Self-contained HTML/CSS diagram (no third-party JS); '
        "colours follow the Gradio theme. Flow reads top→bottom: consumers drive the "
        "per-turn hook loop, whose PRE-hook feeds <b>recall</b> and POST-hook feeds "
        "<b>write</b>; the manual <b>consolidation</b> subsystem and all paths persist "
        "to the shared Postgres + pgvector store.</p></div>"
    )


# ── prose: system summary + the "cron"/lifecycle logic (mirrors ARCHITECTURE.md) ──

_READING_MD = """\
**Diagram in one paragraph.** Three *consumers* drive the engine — the **Chat tab**,
the **Eval harness**, and the **Memory tab**. The Chat tab runs the **per-turn hook
loop** (`engine.run_turn`): a **PRE-hook** recall (`recall_facts`, k=24) built into a
context block and injected into the prompt, a **REPLY** from the LLM shim
(Haiku/Sonnet/Opus, degrading to a recall-only view with no API key), and a
**POST-hook** write (`store_facts` on the `User:…/Agent:…` turn). Recall is a
**scope filter first** (`user_id ∧ agent_type ∧ project_id ∧ is_active`, `session_id`
omitted), then **four candidate legs** — structured/recency (a pool, not a ranked
leg), lexical/fuzzy (`ts_rank_cd`), entity-tag, and vector-ANN (cosine over
MiniLM-384) — of which the last three are fused by **weighted RRF** (K=60, weights
1/1/4). Fused rows are **ranked MATCH → HITS → RECENCY** (recency halves every 7
days), a **recent-reserve** of 3 slots is topped up, the **top-24** is returned, and
a **batched hit-writeback** bumps `hit_count` only on relevance-matched rows. Recall
runs in **two disjoint sections** — OWN (this human + persona) and TEAM (rest of the
project pool, capped at 8). The **write pipeline** extracts facts (gated by default;
a session-close **verbatim** LLM-free path exists), embeds `first_nonblank(summary,
content)` to a 384-vector, computes a `content_tsv`, and inserts one `agent_memory`
row. A separate **consolidation subsystem** (manual/on-demand) groups near-duplicates
(cosine ≥ 0.85) and classifies each group as CONFLICT (→ automatic newest-wins
supersession, keeping `is_active=True`), RESTATEMENT (→ union-merge), or DISTINCT
(untouched); the **manual mutate ops** the Memory tab drives do the stronger,
user-driven variants. Everything persists to a local **Postgres 16 + pgvector**, and
the automatic and manual merges **reuse the same `insert_root_archive_sources`
primitive** so the two paths never fork.
"""

_SUMMARY_MD = """\
### System summary

**Layers**

| Layer | What it is | Where |
|-------|-----------|-------|
| **Store** | Local Postgres 16 + pgvector (docker, host :5433). Four tables: `agent_memory` (`Vector(384)` HNSW-cosine + `content_tsv` GIN + recency index + self-FKs), the per-scope `memory_tag` registry, `memory_tag_link`, and the `memory_consolidation_run` ledger. | `memory/models.py`, `SCHEMA.md` |
| **Recall / write core** | `PostgresMemoryStore` — recall ranking (`_recall_scoped`, `_rrf_fuse`, `record_recall_hits`), the two-section split, and the write path. Ported byte-faithful; SQLite-degradable. | `memory/store.py` |
| **In-process API** | `recall_facts`, `store_facts`, `store_facts_verbatim` + deterministic scope derivation — replaces production's CLI + warm daemon with plain calls. | `memory/recall.py` |
| **Extraction** | One Haiku call/turn → `(summary, tags, occurred_at[, agent_summary])`; gated / two-section / atomic; pluggable/degradable. | `memory/extraction.py` |
| **Embeddings** | Singleton MiniLM-384, ONNX backend (torch-free), pinned. | `memory/embeddings/` |
| **Lifecycle** | Consolidation (near-dup synthesize-merge + conflict supersession) + manual mutate ops (supersede / merge / reset). See below. | `memory/consolidation.py`, `memory/mutate.py` |
| **Consumers** | Eval harness · Gradio UI (Chat / Evaluation / Memory tabs) · the per-turn hook replay. | `eval/harness.py`, `chat/` |

**Scope model.** A memory's coordinates are `(user_id, agent_type, project_id)` with a
`customer_id` for the project pool, derived deterministically as `uuid5` values off
`NS = uuid5(NAMESPACE_DNS, "memory-research.aidachip.com")`. `agent_type` (the persona)
is **the isolation dimension**. **Two-section recall** runs two disjoint scope passes:
**OWN** (`user_id ∧ agent_type ∧ project_id ∧ is_active`) and **TEAM** (the rest of the
project pool, capped at 8). In a single-human demo TEAM is typically empty — a faithful
reflection of the data, not a bug.

**Preserved recall parameters (exact — copied unchanged from production)**

| Parameter | Value | Constant |
|-----------|-------|----------|
| Top-k returned | **24** | `MEMORY_RECALL_K` |
| RRF decay constant K | **60** | `MEMORY_RRF_K` |
| RRF leg weights (fuzzy / tag / vector) | **1.0 / 1.0 / 4.0** | `MEMORY_RRF_W_*` |
| Recency half-life | **7.0 days** | `RECENCY_HALF_LIFE_DAYS` |
| Recent-reserve slots | **3** | `MEMORY_RECALL_RECENT_RESERVE` |
| Candidate cap per leg | **50** | `CANDIDATE_LIMIT` |
| Salience match band | **1000.0** | `SALIENCE_MATCH_BAND` |
| Ranking priority | **MATCH → HITS → RECENCY** | `_recall_scoped` sort key |
| Embedding model / dim | **all-MiniLM-L6-v2 / 384** | `EMBEDDING_MODEL`, `EMBEDDING_DIM` |
| Distance metric | **cosine** (`embedding <=> qvec`) | HNSW `vector_cosine_ops` |
| TEAM-section cap | **8** | `MEMORY_PROJECT_RECALL_K` |

Salience is one inspectable float encoding the same ordering:
`salience = 1000·[rrf>0] + 100·rrf + 10·log1p(hit_count) + recency_decay`, with
`recency_decay = 2^(−age_days / 7)` keyed off **`created_at`** (capture time), never
`occurred_at`.
"""

_LIFECYCLE_MD = """\
### The "crons" / lifecycle logic

The lifecycle has two faces — an **automatic** consolidation pass and the **manual**,
user-invoked mutate ops. They share primitives but differ deliberately in their
`is_active` semantics (below). Everything is **archive, never hard-delete** (the one
exception is the explicit `reset_db` wipe).

**Consolidation — `consolidation.py::consolidate(persona, threshold=0.85)`.** The
single entry-point; resolves the persona's scope, runs the synthesize-merge pass over
that pool, and upserts a durable `MemoryConsolidationRun` ledger row. The pass:

1. Read the pool's **active, current (`valid_to IS NULL`), embedded** corpus, ordered
   `(created_at ASC, id ASC)` for determinism.
2. Drop **never-merge** rows — anything tagged `confidential` or `pinned` is excluded.
3. **Near-dup detection** (`dedup_groups`): greedy, **disjoint** grouping by cosine
   similarity **≥ τ = 0.85** (`DEFAULT_TAU`). Each item lands in at most one group;
   only groups of size ≥ 2 are candidates. Disjointness gives within-pass convergence
   — re-running yields ~0 new merges.
4. For each group, with an LLM and `MEMORY_SUPERSESSION` ON, **classify** it
   (`classify_group`, structured/tool output — never regex-on-prose):
   **CONFLICT** → deterministic newest-wins supersession; **DISTINCT_FACTS** → skip,
   untouched (the safe default; an uncertain/failed classification also defaults here);
   **RESTATEMENT** → union-merge.
5. **Union-merge** (`insert_root_archive_sources`): one Haiku `MERGE_PROMPT` call
   synthesizes a fact that **preserves every distinct detail verbatim** and abstracts
   only connective prose. A group whose merge returns empty is left untouched.

**Manual on-demand — NOT the production background loop.** Production runs
consolidation from an asyncio loop guarded by a Postgres advisory lock + leader guard
+ durable due-gate. That is multi-instance ops infra this replica deliberately strips;
here `consolidate(persona)` is a deterministic, on-demand call (harness or a
"Consolidate now" button) that takes no lock and runs no scheduler. **Degradable
synthesis:** with no LLM, the pass degrades to a deterministic lossless union summary
and skips supersession classification.

**Supersession (automatic conflict) — `supersede_conflict`.** Bi-temporal,
newest-wins, over one value-conflict group. The current fact (latest `valid_from`)
stays as-is; each older member gets `valid_to = winner.valid_from` and
`superseded_by_id = winner.id` — **but keeps `is_active = True`**, so it stays
recallable (the "queryable-A" property). No LLM, no `DELETE`.

**Merge — `mutate.py::merge_memories` + the shared root primitive.** Consolidates
**≥ 2** memories that **share scope** into one new active parent, reusing the **same**
`insert_root_archive_sources` the automatic union-merge uses (no fork): identity
inherited from the newest source, `created_at`/`occurred_at` backdated to the earliest,
union-of-tags provenance, `hit_count = 0`, re-embedded through the same path `write`
uses. Children form a `parent_id` forest and are archived (`is_active = False`).

**Automatic vs manual `is_active` semantics (the key contrast).** All four paths are
**soft** (no hard delete); they differ in whether the affected rows remain recallable:

| Path | Trigger | `is_active` | Recallable after? |
|------|---------|-------------|-------------------|
| **Automatic conflict supersession** | consolidation, group = CONFLICT | **stays `True`** | **Yes** — "queryable-A" |
| **Automatic union-merge** | consolidation, group = RESTATEMENT | **`False`** | No |
| **Manual supersede** (Memory tab) | user retire | **`False`** | No — the explicit retire |
| **Manual merge** (Memory tab) | user merge | **`False`** | No — full audit trail |

The load-bearing difference: **automatic conflict supersession keeps `is_active = True`**
(superseded in the bi-temporal timeline but still returnable), whereas a **manual
`supersede_memory` is a stronger, explicit retire that also sets `is_active = False`**
— which is exactly what makes "recall no longer returns it" true (recall filters on
`is_active`). Manual merge records the **fullest** audit trail (both `parent_id` *and*
`valid_to`/`superseded_by_id`).
"""


def build_system_tab() -> None:
    """Build the System / Architecture tab (call inside a gr.Tab / gr.Blocks). All
    static, self-contained — no engine import, no DB, no API key."""
    gr.Markdown("### System & Architecture")
    gr.Markdown(
        "The canonical system picture of the `memory-research/` replica — the per-turn "
        "hook loop, the four-leg recall pipeline, the write pipeline, the "
        "consolidation/lifecycle subsystem, and the Postgres + pgvector store. Every "
        "constant and lifecycle claim mirrors `docs/design/ARCHITECTURE.md`, verified "
        "against the shipped `memory/*.py`.",
        elem_classes=["mr-subtitle"],
    )
    gr.HTML(value=render_system_diagram())
    gr.Markdown(_READING_MD, elem_classes=["mr-subtitle"])
    gr.Markdown(_SUMMARY_MD)
    gr.Markdown(_LIFECYCLE_MD)
