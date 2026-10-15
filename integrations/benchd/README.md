# Mem++ for Bench'd

This evidence-only adapter implements Bench'd's `setup`, `reset`, `ingest`,
`recall`, and `teardown` interface at harness commit
`bd4824fe5efc095151115c45219c1ef0baa88132`. It uses the existing authenticated
Mem++ Textual Add/Search service. A local adapter or smoke check does not imply
a Bench'd listing, verified receipt, accepted score or leaderboard entry.

## Local setup

Start the [Mem++ service](../../README.md#textual-evaluation-api-aml-preparation) with a dedicated migrated
UTF-8 PostgreSQL/pgvector database, enabled local ONNX MiniLM-384 embeddings and a
`MEMORY_API_TOKEN` of at least 32 printable ASCII characters without whitespace.
The service default is loopback port 8787. Use the same token in the adapter
process. Keep credentials in the environment; no credential file is loaded by
this adapter.

Install the pinned harness into the environment used for the Mem++ checkout:

```sh
python -m pip install 'git+https://github.com/benchdai/harness.git@bd4824fe5efc095151115c45219c1ef0baa88132'
export MEMPP_BENCHD_URL=http://127.0.0.1:8787
export MEMPP_BENCHD_TOP_K=50
# Set MEMORY_API_TOKEN through your environment before running these commands.
python -m integrations.benchd.run_harness adapter validate mempp
python -m integrations.benchd.run_harness run -a mempp -b smoke-memory-v0 --no-judge --out ./benchd-local-smoke
```

Run these commands from the Mem++ checkout. The wrapper registers the adapter
in memory and leaves the upstream source untouched. Other adapters are removed
from this process's registry, so alternative `-a`/`--adapter` values cannot
route to a model-backed builtin. SDK validation and the
retrieval-only smoke run call the local memory service, without model APIs.
The wrapper rejects `--judge` until a separately reviewed integration uses the
existing shared benchmark spending ledger. Bench'd's official protocol locks
the answerer and judge to `openai/gpt-4o-mini`; retrieval-only smoke output is
not a reproduction of that protocol. Organizer listing and verification remain
separate; do not purchase a verification plan through this integration.

## Scope, ingestion and evidence

The pinned harness calls `reset()` for each independent benchmark item. Reset
selects a fresh random world without deleting any existing records. All turns
and ingestion calls within that item share its world; Mem++ organization and
project sharing behavior is preserved. Teardown closes the HTTP client. Stored
worlds remain available for audit and need the database's normal retention
policy. Set up a new adapter for a new run; no automatic resume is provided.

Source content and metadata are retained. User/assistant roles map directly to
Add/Search; a system turn uses the user transport role and explicitly retains
`Original role: system` in its source text. Timezone-aware ISO timestamps are
converted to UTC Unix milliseconds. Missing times remain missing; naive times
are rejected rather than assigned an invented timezone.

An exact repeated ingestion batch is intentionally deduplicated within its
world by its canonical content hash, including after a lost response. Identical
turns inside one batch remain distinct indexed records. If two separate events
have identical content, supply their distinguishing timestamp or metadata.
Invalid or oversized batches fail before transport; the API limit is 4 MiB and
no turn is silently truncated. Recall returns ranked source text with stable
record UUIDs, without generating an answer. Empty evidence returns an empty
string. Authentication, transport and protocol errors propagate to the harness.

Remote endpoints require HTTPS; loopback permits HTTP. Redirects are rejected,
and environment proxy/netrc configuration is not used by the authenticated
client. The configured URL must be the API root origin without embedded
credentials, query parameters or fragments. `MEMPP_BENCHD_TOP_K` accepts 1..100;
record the actual value alongside every run.

## Local tests

With the pinned harness available on `PYTHONPATH` or installed:

```sh
python -m pytest integrations/benchd/test_adapter.py -q
```

These structural tests use a fake transport and no model. Actual HTTP/ONNX/PG
validation should be recorded separately with exact source and engine pins.
When inspecting a local harness manifest, check every trace for adapter errors;
the pinned harness can exit successfully after recording infrastructure failures.
Do not treat its process exit code alone as a valid completed evaluation.
