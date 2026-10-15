# Mem++ for LongMemEval-V2 — local preparation

This native `Memory` implementation targets LongMemEval-V2 commit
`2cc8c540bdb87fe6761629b585e727e1c4704520`. It indexes normalized trajectory states
with Mem++ and returns retrieved source text alongside the corresponding copied
screenshots. Retrieval is guided by query text: neither stored screenshots nor
the optional query image are image-embedded. The native reader receives the
retrieved images. This is an adapter, not a completed benchmark, accepted result
or leaderboard entry.

## Environment

Use a dedicated migrated **UTF-8 PostgreSQL/pgvector** database with `vector(384)`.
Keep database credentials in `MEMORY_DATABASE_URL`, not in a memory config or
saved artifact. No public service or API token is needed: this adapter calls the
engine's reviewed ingestion/search transaction code directly inside the process.
Keep database access private.

Install the pinned Mem++ wheel and the pinned upstream's requirements in a
Python 3.13 environment, plus `httpx` (imported by upstream) and `pytest` for tests:

```sh
python -m pip install 'memory-research @ git+https://github.com/AIDAChip-Inc/mem-plus-plus.git@d952129834f1749c412a04528a4f9138d94a7d74'
python -m pip install -r /absolute/LongMemEval-V2/requirements.txt httpx pytest
export PYTHON_DOTENV_DISABLED=1
export MEMORY_EMBEDDING_BACKEND=onnx
export MEMORY_EMBEDDINGS_ENABLED=1
export MEMORY_RRF_W_FUZZY=1 MEMORY_RRF_W_TAG=1 MEMORY_RRF_W_VECTOR=2
export MEMORY_CANDIDATE_LIMIT=200 MEMORY_LEXICAL_OR=1
# Set MEMORY_DATABASE_URL explicitly and migrate the dedicated database.
export PYTHONPATH=/absolute/LongMemEval-V2
```

The gateway checks the installed engine commit, imported engine Python sources
against the wheel's recorded SHA-256 file hashes,
embedding profile and database schema/encoding. It requires the local model to
load. No generative model is called during insertion or retrieval.

## Build and save without model APIs

Run from the Mem++ checkout. Example `memory_config.json`:

```json
{
  "memory_type": "mempp",
  "memory_params": {
    "trajectories_root_dir": "/absolute/trajectory-data",
    "workspace_dir": "/absolute/local-build-cache",
    "top_k": 10
  }
}
```

The upstream automatically injects runtime paths only for its known baselines;
provide both paths explicitly for this adapter. Each new memory receives a
fresh random scope. All inserted trajectories in that experiment share memory.
Questions using that shared haystack query the same scope; queries never reset
it or ingest question labels. Different native memory instances have different
scopes. The adapter reads no question metadata, gold answers or IDs.

```sh
python -m integrations.longmemeval_v2.build_only \
  --domain enterprise \
  --questions-path /absolute/questions.json \
  --haystack-path /absolute/haystack.json \
  --trajectories-path /absolute/trajectory-data/trajectories.json \
  --memory-config-path /absolute/memory_config.json \
  --output-dir /absolute/new-build-output \
  --skip-evaluation --save-memory
```

This wrapper registers only Mem++ and requires the native build-only flags. It
does not expose reader/judge execution. The pinned native build-only path requires
all questions to share the same ordered haystack. Use actual official inputs
for a future benchmark; synthetic integration checks are not benchmark results.

## Evidence and persistence

Each state retains its trajectory ID, goal, outcome, start URL, step, URL,
thoughts, action and source text as normalized by the pinned native helper.
Record UUIDs are stable within the persisted scope. Identical trajectory retries
are idempotent. Reusing an ID with changed logical content or screenshot bytes
is rejected. PNG, JPEG, WebP and GIF screenshots are validated and copied with
their correct extension; images are not rewritten or symlinked.

Saving verifies database source contents, IDs and vector presence, and copies
screenshots into the artifact with SHA-256 checksums. Loading checks the exact
native memory config, engine pin, image checksums and retained database records.
The scope UUID survives moving the artifact. The artifact's screenshots are
self-contained; the database records must remain available separately. No database
credentials are persisted. Saved/loaded snapshots reject new insertions so later
writes cannot silently change a recorded experiment. Build a new experiment for
new input. Build caches and database worlds require explicit retention/cleanup.

To load in a native Python process, import this adapter before calling the native
factory; no upstream files need modification:

```python
from integrations.longmemeval_v2.mempp import MemppMemory
from memory_modules.memory import load_memory

memory = load_memory("/absolute/saved-memory-state")
context = memory.query("Where are the account settings?")
```

## Validation and remaining evaluation work

```sh
python -m pytest integrations/longmemeval_v2/test_mempp.py -q
```

Unit tests use the actual pinned native factory/trajectory helper and a synthetic
gateway. Actual native build/save/load, ONNX/PG retrieval and reader image-message
format checks are separate integration evidence. These checks do not run the
native context-budget processor, reader, judge, complete question set or LAFS.

A leaderboard submission still needs complete paired web and enterprise runs,
the venue's fixed Qwen3.5-9B reader and GPT-5.2 judge, official context/latency
and LAFS packaging, a reviewed guarded-cost plan, and all three correctness
reviews. Use the official submission form, not a GitHub result issue. Reader or
judge execution must use the existing shared budget ledger; this build wrapper
does not authorize a separate spending pool or paid hosting.
