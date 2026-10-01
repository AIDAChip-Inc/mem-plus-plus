<h1 align="center">
  <img src="docs/brand/mem-plus-plus-header.svg" width="100%" alt="Mem++ — AIDAChip Organization Memory">
</h1>

<p align="center">
  <b>Non-Destructive Memory for Long-Term Organizational LLM Agents</b>
</p>

<p align="center">
  <a href="paper/mempp_iclr2027_submission.pdf">Paper</a>
  ·
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
  <a href="#results">Results</a>
  ·
  <a href="#citation">Citation</a>
  ·
  <a href="SECURITY.md">Security</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL 16 with pgvector">
  <img src="https://img.shields.io/badge/ingest-zero%20LLM%20calls-2ea44f" alt="Zero LLM calls at ingest">
  <img src="https://img.shields.io/badge/memory-append--only-8250df" alt="Append-only memory">
  <a href="paper/mempp_iclr2027_submission.pdf"><img src="https://img.shields.io/badge/paper-ICLR%202027%20(under%20review)-b31b1b" alt="Paper: ICLR 2027, under review"></a>
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
git clone https://github.com/AIDAChip-Inc/mem-plus-plus.git
cd mem-plus-plus
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

### Per-benchmark settings

The paper fixes *k* = 50 everywhere and varies only these settings across benchmarks
(Table 6 of the paper). The last row gives the matching environment variables.

| Setting | OrgMemBench | LoCoMo | LongMemEval<sub>S</sub> |
|---|:---:|:---:|:---:|
| Embedding model | all-MiniLM-L6-v2 | text-embedding-3-small | text-embedding-3-small |
| Embedding size | 384 | 1536 | 1536 |
| Weights (w<sub>lex</sub>, w<sub>tag</sub>, w<sub>vec</sub>) | (1, 1, 4) | (1, 1, 2) | (1, 1, 4) |
| Candidates per index *C* | 50 | 200 | 50 |
| Lexical matching | conjunctive | disjunctive | conjunctive |
| Environment | `MEMORY_EMBEDDING_BACKEND=onnx`<br>`MEMORY_RRF_W_VECTOR=4`<br>`MEMORY_CANDIDATE_LIMIT=50`<br>`MEMORY_LEXICAL_OR=0` | `MEMORY_EMBEDDING_BACKEND=openai`<br>defaults otherwise | `MEMORY_EMBEDDING_BACKEND=openai`<br>`MEMORY_RRF_W_VECTOR=4`<br>`MEMORY_CANDIDATE_LIMIT=50`<br>`MEMORY_LEXICAL_OR=0` |

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

Question counts per category for the three benchmarks used in the paper (Table 5 of the paper):

| OrgMemBench (medium) | Count | LoCoMo | Count | LongMemEval<sub>S</sub> | Count |
|:---|:---:|:---|:---:|:---|:---:|
| C1 Supersession | 15 | Single-hop | 841 | Knowledge update | 72 |
| C2 Provenance | 15 | Multi-hop | 282 | Multi-session | 121 |
| C3 Bi-temporal | 7 | Temporal | 321 | Temporal reasoning | 127 |
| C4 Audit replay | 15 | Open domain | 96 | Single-session user | 64 |
| C5 Justification | 15 | – | – | Single-session assistant | 56 |
| C6 Contradiction | 6 | – | – | Single-session preference | 30 |
| **Total** | **73** | **Total** | **1,540** | **Total** | **470** |

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

`eval.run_longmemeval`, `eval.run_longbench` and `eval.run_membench` take the same flags. The
paper's results are summarized in the next section.

# Results

All numbers in this section are **as reported in the paper** ([`paper/mempp_iclr2027_submission.pdf`](paper/mempp_iclr2027_submission.pdf), Tables 1 to 4 and 7, Figure 4). Scores are LLM-judge scores from a `gpt-4o-mini` judge on a 0 to 100 scale. Within each answerer block, **bold** marks the best result and <ins>underline</ins> the second best, as in the paper.

### OrgMemBench

Performance by question type. *Overall* is weighted by the number of questions in each category and reported with its standard deviation across runs.

**Answerer `gpt-4.1-mini`**

| Method | Supersession | Decision Provenance | Bi-temporal | Audit Replay | Justification Chain | Contradiction | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Context | 0.0 | 27.5 | 0.0 | 33.0 | 22.7 | 8.3 | 17.8 ± 0.1 |
| RAG | 66.1 | **79.7** | **66.7** | 40.9 | 19.6 | **75.4** | 55.0 ± 0.3 |
| Zep | 31.4 | 42.5 | 23.0 | 76.8 | 20.7 | 43.3 | 41.0 ± 0.2 |
| Mem0 | 40.8 | 55.0 | 50.0 | 25.1 | 14.4 | <ins>52.8</ins> | 36.9 ± 0.0 |
| A-Mem | 58.3 | 63.3 | 43.6 | 36.5 | 20.7 | 43.5 | 44.5 ± 0.3 |
| gbrain | 54.8 | 60.7 | 30.3 | 48.5 | **22.7** | 22.9 | 43.2 ± 0.1 |
| Mem++ | **67.7** | <ins>65.9</ins> | 50.0 | **83.0** | <ins>21.5</ins> | 47.2 | **57.6 ± 0.1** |
| Mem<sup>g</sup>++ | <ins>67.2</ins> | 63.9 | <ins>50.6</ins> | <ins>79.7</ins> | 19.8 | 44.4 | <ins>55.9 ± 0.1</ins> |

**Answerer `gpt-4o-mini`**

| Method | Supersession | Decision Provenance | Bi-temporal | Audit Replay | Justification Chain | Contradiction | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Context | 2.5 | 21.1 | 0.0 | 35.7 | 17.0 | 13.9 | 16.8 ± 0.1 |
| RAG | <ins>56.7</ins> | 47.8 | 30.2 | 55.1 | <ins>10.4</ins> | **67.2** | 43.4 ± 0.2 |
| Zep | 27.5 | 38.3 | **34.9** | **74.9** | 7.4 | 29.6 | 36.2 ± 0.3 |
| Mem0 | 32.3 | 47.2 | <ins>33.3</ins> | 32.4 | **11.8** | <ins>62.8</ins> | 33.8 ± 0.1 |
| A-Mem | 42.8 | **60.0** | 28.6 | 21.3 | 4.4 | 41.7 | 32.6 ± 0.3 |
| gbrain | 47.8 | 46.1 | 25.9 | 46.6 | 7.7 | 21.9 | 34.7 ± 0.2 |
| Mem++ | **57.0** | <ins>49.2</ins> | <ins>33.3</ins> | 70.7 | 8.9 | 39.8 | <ins>44.2 ± 0.1</ins> |
| Mem<sup>g</sup>++ | <ins>56.7</ins> | 48.6 | 32.1 | <ins>71.9</ins> | 8.8 | 37.7 | **44.4 ± 0.1** |

### LoCoMo

Performance by question type, with LLM-judge score, F1 and BLEU-1. The baseline numbers come from Nan et al. (2025).

**Answerer `gpt-4.1-mini`**

<table>
<thead>
<tr><th rowspan="2" align="left">Method</th><th colspan="3">Temporal Reasoning</th><th colspan="3">Open Domain</th><th colspan="3">Multi-Hop</th><th colspan="3">Single-Hop</th><th colspan="3">Average</th></tr>
<tr><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th></tr>
</thead>
<tbody>
<tr><td align="left">Full Context</td><td align="center">74.2</td><td align="center">47.5</td><td align="center">40.0</td><td align="center">56.6</td><td align="center">28.4</td><td align="center">22.2</td><td align="center">77.2</td><td align="center">44.2</td><td align="center">33.7</td><td align="center">86.9</td><td align="center">61.4</td><td align="center">53.4</td><td align="center">80.6</td><td align="center">53.3</td><td align="center">45.0</td></tr>
<tr><td align="left">RAG-2048</td><td align="center">66.8</td><td align="center"><ins>50.3</ins></td><td align="center"><ins>40.1</ins></td><td align="center">48.6</td><td align="center"><ins>28.3</ins></td><td align="center">21.7</td><td align="center">67.4</td><td align="center">39.1</td><td align="center">28.9</td><td align="center">82.8</td><td align="center">57.7</td><td align="center">48.1</td><td align="center">74.5</td><td align="center">50.9</td><td align="center">41.3</td></tr>
<tr><td align="left">RAG-4096</td><td align="center">27.4</td><td align="center">22.3</td><td align="center">19.1</td><td align="center">28.8</td><td align="center">17.9</td><td align="center">13.9</td><td align="center">31.7</td><td align="center">20.1</td><td align="center">12.8</td><td align="center">35.9</td><td align="center">25.8</td><td align="center">22.0</td><td align="center">32.9</td><td align="center">23.5</td><td align="center">19.2</td></tr>
<tr><td align="left">Zep</td><td align="center">60.2</td><td align="center">23.9</td><td align="center">20.0</td><td align="center">43.8</td><td align="center">24.2</td><td align="center">19.3</td><td align="center">53.7</td><td align="center">30.5</td><td align="center">20.4</td><td align="center">66.9</td><td align="center">45.5</td><td align="center">40.0</td><td align="center">61.6</td><td align="center">36.9</td><td align="center">30.9</td></tr>
<tr><td align="left">Mem0</td><td align="center">56.9</td><td align="center">39.2</td><td align="center">33.2</td><td align="center">47.9</td><td align="center">23.7</td><td align="center">17.7</td><td align="center">68.2</td><td align="center">40.1</td><td align="center">30.3</td><td align="center">71.4</td><td align="center">48.6</td><td align="center">42.0</td><td align="center">66.3</td><td align="center">43.5</td><td align="center">36.5</td></tr>
<tr><td align="left">A-Mem</td><td align="center">66.7</td><td align="center">40.3</td><td align="center">33.7</td><td align="center">37.5</td><td align="center">13.4</td><td align="center">12.7</td><td align="center">55.7</td><td align="center">30.4</td><td align="center">20.0</td><td align="center">64.0</td><td align="center">45.0</td><td align="center">39.8</td><td align="center">61.4</td><td align="center">39.4</td><td align="center">33.2</td></tr>
<tr><td align="left">Nemori</td><td align="center">77.6</td><td align="center"><b>57.7</b></td><td align="center"><b>50.2</b></td><td align="center"><ins>51.0</ins></td><td align="center">25.8</td><td align="center">19.3</td><td align="center"><b>75.1</b></td><td align="center"><ins>41.7</ins></td><td align="center"><ins>31.9</ins></td><td align="center">84.9</td><td align="center">58.8</td><td align="center">51.5</td><td align="center">79.4</td><td align="center"><b>53.4</b></td><td align="center"><b>45.6</b></td></tr>
<tr><td align="left">Mem++</td><td align="center"><b>81.4</b></td><td align="center">41.7</td><td align="center">33.5</td><td align="center"><b>53.0</b></td><td align="center"><b>29.0</b></td><td align="center"><ins>22.1</ins></td><td align="center"><ins>74.8</ins></td><td align="center"><b>41.9</b></td><td align="center"><b>32.0</b></td><td align="center"><b>87.0</b></td><td align="center"><b>62.1</b></td><td align="center"><ins>54.8</ins></td><td align="center"><b>81.5</b></td><td align="center"><ins>52.1</ins></td><td align="center">44.2</td></tr>
<tr><td align="left">Mem<sup>g</sup>++</td><td align="center"><ins>81.1</ins></td><td align="center">41.8</td><td align="center">33.6</td><td align="center">50.5</td><td align="center">27.9</td><td align="center"><b>22.8</b></td><td align="center">71.8</td><td align="center">41.2</td><td align="center"><b>32.0</b></td><td align="center"><ins>86.4</ins></td><td align="center"><ins>62.0</ins></td><td align="center"><b>55.0</b></td><td align="center"><ins>80.4</ins></td><td align="center">51.9</td><td align="center"><ins>44.3</ins></td></tr>
</tbody>
</table>

**Answerer `gpt-4o-mini`**

<table>
<thead>
<tr><th rowspan="2" align="left">Method</th><th colspan="3">Temporal Reasoning</th><th colspan="3">Open Domain</th><th colspan="3">Multi-Hop</th><th colspan="3">Single-Hop</th><th colspan="3">Average</th></tr>
<tr><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th><th>LLM</th><th>F1</th><th>BLEU</th></tr>
</thead>
<tbody>
<tr><td align="left">Full Context</td><td align="center">56.2</td><td align="center">44.1</td><td align="center">36.1</td><td align="center">48.6</td><td align="center">24.5</td><td align="center">17.2</td><td align="center">66.8</td><td align="center">35.4</td><td align="center">26.1</td><td align="center">83.0</td><td align="center">53.1</td><td align="center">44.7</td><td align="center">72.3</td><td align="center">46.2</td><td align="center">37.8</td></tr>
<tr><td align="left">RAG-2048</td><td align="center">62.6</td><td align="center">47.5</td><td align="center">38.2</td><td align="center">45.2</td><td align="center">23.4</td><td align="center">18.1</td><td align="center">63.0</td><td align="center">35.7</td><td align="center">24.6</td><td align="center">78.5</td><td align="center">53.6</td><td align="center">43.0</td><td align="center">70.3</td><td align="center">47.2</td><td align="center">37.1</td></tr>
<tr><td align="left">RAG-4096</td><td align="center">22.8</td><td align="center">18.4</td><td align="center">15.2</td><td align="center">35.5</td><td align="center">17.6</td><td align="center">15.6</td><td align="center">31.1</td><td align="center">18.6</td><td align="center">11.7</td><td align="center">33.0</td><td align="center">23.9</td><td align="center">19.2</td><td align="center">30.7</td><td align="center">21.4</td><td align="center">16.8</td></tr>
<tr><td align="left">Zep</td><td align="center">58.9</td><td align="center">44.8</td><td align="center">38.1</td><td align="center">39.6</td><td align="center">22.9</td><td align="center">15.7</td><td align="center">50.5</td><td align="center">27.5</td><td align="center">19.3</td><td align="center">63.2</td><td align="center">39.7</td><td align="center">33.7</td><td align="center">58.5</td><td align="center">37.5</td><td align="center">30.9</td></tr>
<tr><td align="left">Mem0</td><td align="center">50.4</td><td align="center">44.4</td><td align="center">37.6</td><td align="center">40.6</td><td align="center"><b>27.1</b></td><td align="center"><b>19.4</b></td><td align="center">60.3</td><td align="center">34.3</td><td align="center">25.2</td><td align="center">68.1</td><td align="center">44.4</td><td align="center">37.7</td><td align="center">61.3</td><td align="center">41.5</td><td align="center">34.2</td></tr>
<tr><td align="left">A-Mem</td><td align="center">54.2</td><td align="center">38.1</td><td align="center">33.8</td><td align="center">22.9</td><td align="center">9.0</td><td align="center">8.6</td><td align="center">43.6</td><td align="center">24.0</td><td align="center">18.8</td><td align="center">58.2</td><td align="center">35.6</td><td align="center">29.2</td><td align="center">52.5</td><td align="center">32.4</td><td align="center">27.0</td></tr>
<tr><td align="left">Nemori</td><td align="center">71.0</td><td align="center"><ins>56.7</ins></td><td align="center"><b>46.6</b></td><td align="center">44.8</td><td align="center">20.8</td><td align="center">15.1</td><td align="center">65.3</td><td align="center">36.5</td><td align="center"><ins>25.6</ins></td><td align="center">82.1</td><td align="center">54.4</td><td align="center">43.2</td><td align="center">74.4</td><td align="center">49.5</td><td align="center">38.5</td></tr>
<tr><td align="left">Mem++</td><td align="center"><b>76.9</b></td><td align="center"><b>56.9</b></td><td align="center"><ins>46.5</ins></td><td align="center"><b>47.3</b></td><td align="center">24.2</td><td align="center">17.8</td><td align="center"><b>68.5</b></td><td align="center"><b>39.0</b></td><td align="center"><b>27.5</b></td><td align="center"><ins>84.0</ins></td><td align="center"><b>59.7</b></td><td align="center"><b>47.9</b></td><td align="center"><b>77.4</b></td><td align="center"><b>53.1</b></td><td align="center"><b>42.0</b></td></tr>
<tr><td align="left">Mem<sup>g</sup>++</td><td align="center"><ins>75.6</ins></td><td align="center">55.3</td><td align="center">44.5</td><td align="center"><ins>46.2</ins></td><td align="center"><ins>24.6</ins></td><td align="center"><ins>18.8</ins></td><td align="center"><ins>65.8</ins></td><td align="center"><ins>37.2</ins></td><td align="center">25.3</td><td align="center"><b>84.1</b></td><td align="center"><ins>58.9</ins></td><td align="center"><ins>47.3</ins></td><td align="center"><ins>76.7</ins></td><td align="center"><ins>52.1</ins></td><td align="center"><ins>41.0</ins></td></tr>
</tbody>
</table>

### LongMemEval<sub>S</sub>

LLM-judge accuracy by question type.

**Answerer `gpt-4o-mini`**

| Question type | Full Context | Zep | Nemori | Mem++ | Mem<sup>g</sup>++ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Single-session preference | 6.7 | 20.0 | <ins>46.7</ins> | <ins>46.7</ins> | **50.0** |
| Single-session assistant | 89.3 | 80.4 | 83.9 | <ins>94.6</ins> | **96.4** |
| Temporal reasoning | 42.1 | **62.4** | <ins>61.7</ins> | 56.7 | 56.7 |
| Multi-session | 38.3 | 57.9 | 51.1 | **63.4** | <ins>62.0</ins> |
| Knowledge update | 78.2 | 83.3 | 61.5 | <ins>84.3</ins> | **85.2** |
| Single-session user | 78.6 | <ins>92.9</ins> | 88.6 | **98.4** | **98.4** |
| **Average** | 55.0 | 68.0 | 64.2 | <ins>72.2</ins> | **72.4** |

**Answerer `gpt-4.1-mini`**

| Question type | Full Context | Zep | Nemori | Mem++ | Mem<sup>g</sup>++ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Single-session preference | 16.7 | 22.5 | **86.7** | 47.8 | <ins>53.3</ins> |
| Single-session assistant | **98.2** | 83.1 | 92.9 | <ins>94.6</ins> | <ins>94.6</ins> |
| Temporal reasoning | 60.2 | 64.5 | **72.2** | 68.5 | <ins>69.0</ins> |
| Multi-session | 51.1 | **57.6** | 55.6 | 56.1 | <ins>56.8</ins> |
| Knowledge update | 76.9 | <ins>83.1</ins> | 79.5 | **89.9** | **89.9** |
| Single-session user | 85.7 | <ins>96.3</ins> | 90.0 | **100.0** | **100.0** |
| **Average** | 65.6 | 69.4 | 74.6 | <ins>74.7</ins> | **75.3** |

### Ablation

Ablation with `claude-sonnet-4-6` as the answerer and the same `gpt-4o-mini` judge. Parentheses give the change from Mem++.

| Variant | OrgMemBench | LoCoMo | LongMemEval<sub>S</sub> |
|:---|:---:|:---:|:---:|
| Mem++ | 57.0 | 85.7 | 87.9 |
| w/o vector leg | 21.3 (−35.7) | 76.7 (−9.1) | 11.9 (−76.0) |
| w/o lexical leg | 57.2 (+0.2) | 85.2 (−0.5) | 88.7 (+0.7) |
| w/o tag leg | 55.0 (−2.0) | – | – |
| w/ consolidation | 55.6 (−1.3) | 85.4 (−0.3) | 86.9 (−1.1) |
| w/ fact index | 55.2 (−1.7) | 83.1 (−2.6) | – |
| Mem<sup>g</sup>++ | 55.3 (−1.7) | 84.8 (−0.9) | 86.0 (−2.0) |

### Retrieval depth *k* on OrgMemBench

<p align="center">
  <img src="docs/figures/topk_sensitivity.png" width="100%" alt="OrgMemBench score of Mem++ against the number of retrieved rows k, for gpt-4o-mini and gpt-4.1-mini answerers, with Full Context as a dashed baseline">
</p>

OrgMemBench score of Mem++ as the number of retrieved rows *k* varies. The dashed line is Full Context, and the annotated point is *k* = 50, the setting used in the main experiments. Retrieved tokens are summed over all questions and counted before the 40,000-character cut.

| *k* | gpt-4o-mini | gpt-4.1-mini | Retrieved tokens |
|:---:|:---:|:---:|:---:|
| 5 | 36.57 | 43.99 | 0.20M |
| 10 | 45.10 | 50.56 | 0.41M |
| 15 | 47.16 | 54.44 | 0.61M |
| 30 | 45.94 | 56.49 | 1.22M |
| 50 | 44.23 | 57.00 | 2.01M |
| 70 | 48.67 | 56.70 | 2.77M |
| 100 | 44.89 | 56.54 | 3.63M |
| Full Context | 16.85 | 21.52 | 18.68M |

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
├── paper/                   the paper (ICLR 2027 submission, under review)
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
  note   = {Under review at ICLR 2027}
}
```

# License

Apache 2.0 — see the [LICENSE](LICENSE) file for details.
