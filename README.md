<h1 align="center">Mem++</h1>

<p align="center">
  <b>Non-Destructive Memory for Long-Term Organizational LLM Agents</b>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a>
  ·
  <a href="#how-it-works">How it works</a>
  ·
  <a href="#basic-usage">Usage</a>
  ·
  <a href="#configuration">Configuration</a>
  ·
  <a href="#evaluation">Evaluation</a>
  ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL 16 with pgvector">
  <img src="https://img.shields.io/badge/ingest-zero%20LLM%20calls-2ea44f" alt="Zero LLM calls at ingest">
  <img src="https://img.shields.io/badge/memory-append--only-8250df" alt="Append-only memory">
  <img src="https://img.shields.io/badge/paper-under%20review-lightgrey" alt="Paper under review">
</p>

---

# Introduction

**Mem++** is a memory engine for LLM agents that work over an organization's record: the
Slack threads, emails, meeting notes, tickets and documents that many people write over
months and years, and that revise, supersede and contradict one another.

Most agent-memory systems are designed for *conversational* memory, where one user narrates
their own facts to an assistant. They distil each exchange into a handful of facts at write
time, and whatever the extractor did not keep is gone for good. That trade works when a
single narrator updates their own story. It fails when many authors write over each other,
because the questions an organization actually asks need the original records rather than a
summary of them: *what did we believe on 12 March, what did this decision replace, and which
document authorized it?*

Mem++ takes the opposite trade. It keeps every artifact verbatim and dated, spends **no LLM
calls when writing**, and does its understanding at read time. A question is answered through
a date filter, three retrieval indexes fused by weighted rank, a few slots reserved for the
most recent evidence, and a dated, attributed evidence block handed to the answering model.
Nothing is overwritten, so every version stays retrievable.

<p align="center">
  <img src="docs/figures/conversational_vs_organizational.png" width="100%" alt="Conversational memory distils a single narrator's facts at ingest and loses the originals; organizational memory keeps every record from many authors and lets the question do the choosing">
  <br>
  <em>Conversational memory distils one narrator's facts at write time. Organizational memory has to keep every author's record, because the question decides what matters.</em>
</p>

## Key features

**Core capabilities**

- **Non-destructive by design.** The store is append-only. Superseded and conflicting records
  are marked, never deleted, so any earlier state of the record can be replayed.
- **Zero LLM calls at ingest.** The verbatim write path embeds and indexes each record without
  calling a language model, so ingest cost does not grow with model pricing.
- **As-of queries.** An event-time predicate, `occurred_at ≤ θ`, sits in the base query, so a
  question can be asked of the record exactly as it stood on any past date.
- **Multi-index retrieval.** Lexical full-text search, dense vectors and entity tags each rank
  the candidates independently, and weighted Reciprocal Rank Fusion combines them.
- **Recency slots.** A few of the top-k slots are reserved for the latest-dated records in
  scope, so the newest version of a fact is always in view next to the most relevant ones.
- **Dated, attributed evidence.** Every retrieved record reaches the answering model with its
  date and its contributor.
- **Optional consolidation.** Near-duplicate records can be grouped and labelled as a
  conflict, a restatement or distinct. Originals are archived, not removed.
- **Pluggable embeddings.** A local, CPU-only ONNX MiniLM-384 embedder by default, or OpenAI
  `text-embedding-3-small`.

**Where it fits**

- **Organizational assistants** that answer from Slack, email, meetings, tickets and documents.
- **Audit and compliance**, where the question is what was on record at a given date.
- **Decision provenance**: who decided, when, which alternatives were weighed, and what the
  decision replaced.
- **Incident review**: how the understanding of a root cause changed over time.

# How it works

<p align="center">
  <img src="docs/figures/architecture.png" width="100%" alt="Mem++ architecture in three stages: memory building and storing, memory activation via multi-index retrieval, and dated evidence presentation">
</p>

Mem++ runs in three stages.

**1. Memory building and storing.** Exact repeats within a scope are skipped. Every other
artifact becomes one memory record holding its full text, contributor, event date and
embedding. Records land in an append-only store and are indexed three ways: lexically
(PostgreSQL full-text search ranked by `ts_rank_cd`), by vector (pgvector) and by entity tag.

**2. Memory activation via multi-index retrieval.** A query arrives with an optional as-of
date θ. The date filter restricts the pool to active records with `t ≤ θ`. The lexical, vector
and tag indexes each return their own ranking, and weighted Reciprocal Rank Fusion combines
the three. The final shortlist of *k* records takes *k − 3* by fused rank and reserves 3 slots
for the latest-dated records.

**3. Dated evidence presentation.** The shortlist is rendered as a dated, attributed evidence
block, one entry per record, and passed to the answering model.

The boxes in the figure follow one convention:

| Box | Meaning |
|---|---|
| Solid | Core Mem++. No language-model call anywhere on the write path. |
| Dotted | Optional consolidation. Groups near-duplicates and labels what changed; originals are archived. |
| Dashed | The Mem<sup>g</sup>++ variant, which adds an entity graph built at write time and appends up to five graph triples after the evidence rows. |
| <kbd>LLM</kbd> | A language-model call. |

> [!NOTE]
> This repository contains the core Mem++ engine and the optional consolidation pass. The
> Mem<sup>g</sup>++ entity-graph variant is a research extension and is not packaged in this
> release.

# Quickstart

| | Docker | Docker-free (macOS) |
|---|---|---|
| **Best for** | Any OS, running the engine and the evaluation harness | A self-contained local demo |
| **Database** | PostgreSQL 16 + pgvector container on port 5433 | Project-local PostgreSQL 17 cluster over a unix socket |
| **Start** | `docker compose up -d db` | `./run_demo.sh` |

Both paths use [`uv`](https://docs.astral.sh/uv/) and Python 3.13 or newer. No GPU is needed.

```bash
git clone https://github.com/yehiahmad/mempp.git
cd mempp
uv sync --extra llm --extra openai
```

### Docker

```bash
docker compose up -d db
export MEMORY_DATABASE_URL=postgresql+psycopg2://memory:memory@localhost:5433/memory_research
uv run alembic upgrade head
```

The engine reads `MEMORY_DATABASE_URL` from the environment; it does not load `.env` on its
own. `.env.example` documents every connection option.

### Docker-free (macOS)

```bash
brew install postgresql@17 pgvector
uv sync --extra chat
./run_demo.sh          # create the local cluster, migrate, and launch the demo on http://127.0.0.1:7860
./run_demo.sh stop     # stop the cluster; data is kept
./run_demo.sh reset    # wipe the local database (asks first)
```

Install `postgresql@17` specifically, since that Homebrew formula is the one that ships
pgvector. Keep the checkout at a short path: the cluster listens on a unix socket inside the
project, and operating systems cap socket path length.

### Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `llm` | Anthropic SDK | LLM fact extraction (`store_facts`), consolidation, Claude answer and judge models |
| `openai` | OpenAI SDK | The `text-embedding-3-small` backend, GPT answer and judge models |
| `chat` | Gradio | The local demo |
| `test` | pytest | The test suite |

Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your environment for the extras you use. The
core write and read paths need neither.

# Basic usage

Write records verbatim, then recall them. The first call downloads the ONNX MiniLM model.

```python
from memory import store_facts_verbatim, recall_facts

# Write: each record is stored as-is with its event date. No LLM call is made.
store_facts_verbatim("ops-agent", [
    {"summary": "VPN runs on Vendor A.",       "tags": ["vpn", "vendor-a"], "occurred_at": "2023-12-28"},
    {"summary": "We keep Vendor A this year.", "tags": ["vpn", "vendor-a"], "occurred_at": "2024-01-09"},
    {"summary": "Board approved Vendor B.",    "tags": ["vpn", "vendor-b"], "occurred_at": "2025-03-27"},
    {"summary": "Vendor B is now live.",       "tags": ["vpn", "vendor-b"], "occurred_at": "2025-09-09"},
])

# Read: date filter, three indexes, weighted RRF and recency slots.
for row in recall_facts("ops-agent", "Which VPN vendor do we use?", k=10):
    print(row["occurred_at"], row["summary"], row.get("by"))
```

Each fact may be a dict with `summary`, `tags` and `occurred_at`, a
`(summary, tags, occurred_at)` tuple, or a bare string. `recall_facts` returns plain dicts with
`summary`, `occurred_at`, `hit_count` and, when known, `by`, the record's contributors.

### Asking the record as of a past date

The store-level API exposes the as-of predicate directly.

```python
import uuid
from datetime import datetime, timezone

from memory.db import get_session
from memory.store import PostgresMemoryStore

user, project = uuid.uuid4(), uuid.uuid4()
store = PostgresMemoryStore(get_session())

store.write(user_id=user, agent_type="ops-agent", project_id=project,
            content="Board approved Vendor B.", tags=["vpn"],
            occurred_at=datetime(2025, 3, 27, tzinfo=timezone.utc))

# Only records dated on or before 1 June 2024 are eligible, so Vendor B is not yet on record.
as_of = store.recall(user_id=user, agent_type="ops-agent", project_id=project,
                     query="Which VPN vendor do we use?", k=10,
                     occurred_before=datetime(2024, 6, 1, tzinfo=timezone.utc))
```

### Consolidation

Consolidation is manual and on demand; nothing runs in the background. It needs the `llm`
extra and `ANTHROPIC_API_KEY`.

```python
from memory.consolidation import consolidate

report = consolidate("ops-agent", threshold=0.85)
# {"run_id", "pools_merged", "memories_before", "memories_after", "superseded"}
```

Records whose embeddings reach a cosine similarity of at least `threshold` are grouped. A
language model labels each group as a **conflict** (the older version is marked replaced), a
**restatement** (merged, with the originals archived) or **distinct** (left untouched).

# Demo

The Gradio demo runs the engine end to end on your machine.

```bash
uv sync --extra chat
./run_demo.sh                        # Docker-free, or with MEMORY_DATABASE_URL already set:
uv run python -m chat.app            # http://127.0.0.1:7860
```

| Tab | What it does |
|---|---|
| **Chat** | Talk to an assistant that writes to and recalls from Mem++ on every turn. |
| **Memory** | Browse and search stored records, read-only, by keyword or meaning, with filters. |
| **Evaluation** | Run a benchmark against the real engine from the browser. |
| **System** | The architecture diagram, a system summary and the consolidation lifecycle. |

# Configuration

Every setting is an environment variable read by `memory/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_DATABASE_URL` | *(required)* | SQLAlchemy DSN for PostgreSQL with pgvector. |
| `MEMORY_EMBEDDING_BACKEND` | `onnx` | `onnx` for the local MiniLM-384 embedder, `openai` for the OpenAI API. |
| `MEMORY_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model, used when the backend is `openai`. |
| `MEMORY_EMBEDDING_DIM` | `1536` | Dimension of the OpenAI embedding. |
| `MEMORY_EMBEDDINGS_ENABLED` | `1` | Turn the vector index and write-time embedding on or off. |
| `MEMORY_RRF_W_FUZZY` | `1.0` | Fusion weight of the lexical index. |
| `MEMORY_RRF_W_TAG` | `1.0` | Fusion weight of the entity-tag index. |
| `MEMORY_RRF_W_VECTOR` | `2.0` | Fusion weight of the vector index. |
| `MEMORY_CANDIDATE_LIMIT` | `200` | Candidates each index may contribute before fusion. |
| `MEMORY_LEXICAL_OR` | `1` | `1` matches any query term (disjunctive); `0` requires all terms (conjunctive). |
| `MEMORY_USER` | OS login | Overrides the login that scopes a user's memories. |

Fixed constants: Reciprocal Rank Fusion smoothing of 60, 3 recency slots, a 7-day recency
half-life, and a default `k` of 24 for `recall_facts`.

# Evaluation

<p align="center">
  <img src="docs/figures/orgmembench.png" width="100%" alt="OrgMemBench: an organizational memory benchmark of 443 artifacts over 18 months, with 73 questions across six capabilities">
</p>

Mem++ is evaluated on **OrgMemBench**, a benchmark for organizational memory: 443 artifacts in
157 threads spanning 18 months, and 73 questions across six capabilities.

| Code | Capability | What it tests |
|---|---|---|
| C1 | Supersession | The current value of a fact, what it replaced, when and why. |
| C2 | Decision provenance | Who decided, when, which alternatives were weighed, and the deciding rationale. |
| C3 | Bi-temporal (as-of) | What the organization believed as of a past date, as distinct from now. |
| C4 | Audit replay | Reconstruct the knowledge state at a past date and flag what has since changed. |
| C5 | Justification chain | The evidence behind a conclusion, each item classed as direct testimony or inference. |
| C6 | Contradiction | Detect that two artifacts conflict, report both sides, and whether it was resolved. |

OrgMemBench is distributed separately from this repository.

The harness in `eval/` also runs the conversational memory benchmarks **LoCoMo**,
**LongMemEval**, **LongBench** and **MemBench**. Benchmark data is never committed; fetch it
first.

```bash
uv run python datasets/fetch.py locomo            # or: longmemeval | longbench | membench | all

# Smoke test with no database and no API key
uv run python -m eval.run_locomo --stub --limit 1

# Full run: Mem++ at k=50, GPT answerer, GPT judge
mkdir -p runs
uv run python -m eval.run_locomo --k 50 \
    --answer-model gpt-4o-mini --judge-model gpt-4o-mini \
    --workers 4 --out runs/locomo_k50.json
```

| Flag | Effect |
|---|---|
| `--k` | Records recalled per question. |
| `--answer-model`, `--judge-model` | `haiku`, `sonnet`, `opus`, or any raw model id; ids starting with `gpt` route to OpenAI. |
| `--recall-only` | Retrieval metrics only, with no answer or judge calls. |
| `--fraction`, `--limit` | Evaluate a deterministic fraction of questions, or cap the number of conversations. |
| `--reuse-ingest` | Skip ingest for scopes that already hold memories. |
| `--stub` | In-memory backend with echo answer and judge, for wiring checks. |

`eval.run_longmemeval`, `eval.run_longbench` and `eval.run_membench` take the same flags. Results
are reported in the accompanying paper; this repository ships no results.

# Project structure

```
mempp/
├── memory/                  the engine
│   ├── store.py             write path, multi-index recall, weighted RRF, recency slots
│   ├── recall.py            public API: store_facts, store_facts_verbatim, recall_facts
│   ├── consolidation.py     optional near-duplicate grouping and conflict labelling
│   ├── extraction.py        optional LLM fact extraction
│   ├── embeddings/          ONNX MiniLM-384 and OpenAI backends
│   ├── models.py            SQLAlchemy schema
│   ├── mutate.py            supersede and merge operations
│   ├── browse.py            read-only inspection
│   └── config.py            every environment setting
├── eval/                    evaluation harness, benchmark adapters, answer and judge models
├── chat/                    Gradio demo
├── alembic/                 database migrations
├── datasets/fetch.py        benchmark downloader
├── tests/, tests_engine/    test suites
├── docs/figures/            figures used in this README
├── docker-compose.yml       PostgreSQL 16 + pgvector
└── run_demo.sh              Docker-free local cluster and demo
```

# Testing

```bash
uv sync --extra test
uv run pytest
```

End-to-end tests expect a migrated PostgreSQL reachable through `MEMORY_DATABASE_URL`.

# Citation

If you use Mem++ or OrgMemBench, please cite:

```bibtex
@misc{mempp2026,
  title  = {Mem++: Non-Destructive Memory for Long-Term Organizational LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

# License

This repository is private and has not yet been released under an open-source license. All
rights reserved.
